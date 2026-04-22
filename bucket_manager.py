# ============================================================
# Module: Memory Bucket Manager (bucket_manager.py)
# 模块：记忆桶管理器 — Supabase 版本
# ============================================================

import os
import math
import logging
from datetime import datetime
from typing import Optional

from rapidfuzz import fuzz
from supabase import create_client, Client

from utils import generate_bucket_id, sanitize_name, now_iso

logger = logging.getLogger("ombre_brain.bucket")


def _row_to_bucket(row: dict) -> dict:
    """Convert a Supabase DB row to the bucket dict format used by the rest of the code."""
    metadata = {
        "id": row["id"],
        "name": row.get("name", row["id"]),
        "tags": row.get("tags") or [],
        "domain": row.get("domain") or ["未分类"],
        "valence": row.get("valence", 0.5),
        "arousal": row.get("arousal", 0.3),
        "importance": row.get("importance", 5),
        "type": row.get("bucket_type", "dynamic"),
        "created": row.get("created", ""),
        "last_active": row.get("last_active", ""),
        "activation_count": row.get("activation_count", 0),
        "resolved": row.get("resolved", False),
        "pinned": row.get("pinned", False),
        "protected": row.get("protected", False),
        "digested": row.get("digested", False),
    }
    if row.get("model_valence") is not None:
        metadata["model_valence"] = row["model_valence"]
    return {
        "id": row["id"],
        "metadata": metadata,
        "content": row.get("content", ""),
        "path": None,  # No file path in DB mode
    }


class BucketManager:
    def __init__(self, config: dict, embedding_engine=None):
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        self.supabase: Client = create_client(url, key)

        self.fuzzy_threshold = config.get("matching", {}).get("fuzzy_threshold", 50)
        self.max_results = config.get("matching", {}).get("max_results", 5)

        scoring = config.get("scoring_weights", {})
        self.w_topic = scoring.get("topic_relevance", 4.0)
        self.w_emotion = scoring.get("emotion_resonance", 2.0)
        self.w_time = scoring.get("time_proximity", 1.5)
        self.w_importance = scoring.get("importance", 1.0)
        self.content_weight = scoring.get("content_weight", 1.0)

        self.embedding_engine = embedding_engine

    # ---------------------------------------------------------
    # Create
    # ---------------------------------------------------------
    async def create(
        self,
        content: str,
        tags: list = None,
        importance: int = 5,
        domain: list = None,
        valence: float = 0.5,
        arousal: float = 0.3,
        bucket_type: str = "dynamic",
        name: str = None,
        pinned: bool = False,
        protected: bool = False,
    ) -> str:
        bucket_id = generate_bucket_id()
        bucket_name = sanitize_name(name) if name else bucket_id

        if bucket_type == "feel":
            domain = domain if domain is not None else []
        else:
            domain = domain or ["未分类"]
        tags = tags or []

        if pinned or protected:
            importance = 10
        if pinned and bucket_type != "permanent":
            bucket_type = "permanent"

        row = {
            "id": bucket_id,
            "name": bucket_name,
            "content": content,
            "tags": tags,
            "domain": domain,
            "valence": max(0.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "importance": max(1, min(10, importance)),
            "bucket_type": bucket_type,
            "created": now_iso(),
            "last_active": now_iso(),
            "activation_count": 0,
            "resolved": False,
            "pinned": pinned,
            "protected": protected,
            "digested": False,
        }

        self.supabase.table("buckets").insert(row).execute()
        logger.info(f"Created bucket: {bucket_id} ({bucket_name})")
        return bucket_id

    # ---------------------------------------------------------
    # Read
    # ---------------------------------------------------------
    async def get(self, bucket_id: str) -> Optional[dict]:
        if not bucket_id:
            return None
        res = self.supabase.table("buckets").select("*").eq("id", bucket_id).execute()
        if res.data:
            return _row_to_bucket(res.data[0])
        return None

    # ---------------------------------------------------------
    # Update
    # ---------------------------------------------------------
    async def update(self, bucket_id: str, **kwargs) -> bool:
        res = self.supabase.table("buckets").select("pinned, protected").eq("id", bucket_id).execute()
        if not res.data:
            return False
        row = res.data[0]
        is_pinned = row.get("pinned", False) or row.get("protected", False)

        updates = {"last_active": now_iso()}

        if "content" in kwargs:
            updates["content"] = kwargs["content"]
        if "tags" in kwargs:
            updates["tags"] = kwargs["tags"]
        if "importance" in kwargs and not is_pinned:
            updates["importance"] = max(1, min(10, int(kwargs["importance"])))
        if "domain" in kwargs:
            updates["domain"] = kwargs["domain"]
        if "valence" in kwargs:
            updates["valence"] = max(0.0, min(1.0, float(kwargs["valence"])))
        if "arousal" in kwargs:
            updates["arousal"] = max(0.0, min(1.0, float(kwargs["arousal"])))
        if "name" in kwargs:
            updates["name"] = sanitize_name(kwargs["name"])
        if "resolved" in kwargs:
            updates["resolved"] = bool(kwargs["resolved"])
        if "pinned" in kwargs:
            updates["pinned"] = bool(kwargs["pinned"])
            if kwargs["pinned"]:
                updates["importance"] = 10
                updates["bucket_type"] = "permanent"
        if "digested" in kwargs:
            updates["digested"] = bool(kwargs["digested"])
        if "model_valence" in kwargs:
            updates["model_valence"] = max(0.0, min(1.0, float(kwargs["model_valence"])))

        self.supabase.table("buckets").update(updates).eq("id", bucket_id).execute()
        logger.info(f"Updated bucket: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------
    async def delete(self, bucket_id: str) -> bool:
        self.supabase.table("buckets").delete().eq("id", bucket_id).execute()
        logger.info(f"Deleted bucket: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Touch (refresh activation)
    # ---------------------------------------------------------
    async def touch(self, bucket_id: str) -> None:
        res = self.supabase.table("buckets").select("activation_count, created").eq("id", bucket_id).execute()
        if not res.data:
            return
        row = res.data[0]
        new_count = (row.get("activation_count") or 0) + 1
        self.supabase.table("buckets").update({
            "last_active": now_iso(),
            "activation_count": new_count,
        }).eq("id", bucket_id).execute()

        try:
            reference_time = datetime.fromisoformat(str(row.get("created", "")))
            await self._time_ripple(bucket_id, reference_time)
        except Exception:
            pass

    async def _time_ripple(self, source_id: str, reference_time: datetime, hours: float = 48.0) -> None:
        try:
            all_buckets = await self.list_all(include_archive=False)
        except Exception:
            return
        rippled = 0
        for bucket in all_buckets:
            if rippled >= 5:
                break
            if bucket["id"] == source_id:
                continue
            meta = bucket["metadata"]
            if meta.get("pinned") or meta.get("protected") or meta.get("type") in ("permanent", "feel"):
                continue
            try:
                created = datetime.fromisoformat(str(meta.get("created", "")))
                delta_hours = abs((reference_time - created).total_seconds()) / 3600
            except Exception:
                continue
            if delta_hours <= hours:
                current_count = meta.get("activation_count", 1)
                self.supabase.table("buckets").update({
                    "activation_count": round(current_count + 0.3, 1)
                }).eq("id", bucket["id"]).execute()
                rippled += 1

    # ---------------------------------------------------------
    # Search (multi-dimensional, same logic as original)
    # ---------------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = None,
        domain_filter: list = None,
        query_valence: float = None,
        query_arousal: float = None,
    ) -> list:
        if not query or not query.strip():
            return []

        limit = limit or self.max_results
        all_buckets = await self.list_all(include_archive=False)
        if not all_buckets:
            return []

        if domain_filter:
            filter_set = {d.lower() for d in domain_filter}
            candidates = [
                b for b in all_buckets
                if {d.lower() for d in b["metadata"].get("domain", [])} & filter_set
            ]
            if not candidates:
                candidates = all_buckets
        else:
            candidates = all_buckets

        if self.embedding_engine and self.embedding_engine.enabled:
            try:
                vector_results = await self.embedding_engine.search_similar(query, top_k=50)
                if vector_results:
                    vector_ids = {bid for bid, _ in vector_results}
                    emb_candidates = [b for b in candidates if b["id"] in vector_ids]
                    if emb_candidates:
                        candidates = emb_candidates
            except Exception as e:
                logger.warning(f"Embedding pre-filter failed: {e}")

        scored = []
        for bucket in candidates:
            meta = bucket.get("metadata", {})
            try:
                topic_score = self._calc_topic_score(query, bucket)
                emotion_score = self._calc_emotion_score(query_valence, query_arousal, meta)
                time_score = self._calc_time_score(meta)
                importance_score = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                total = (
                    topic_score * self.w_topic
                    + emotion_score * self.w_emotion
                    + time_score * self.w_time
                    + importance_score * self.w_importance
                )
                weight_sum = self.w_topic + self.w_emotion + self.w_time + self.w_importance
                normalized = (total / weight_sum) * 100 if weight_sum > 0 else 0

                if normalized >= self.fuzzy_threshold:
                    if meta.get("resolved", False):
                        normalized *= 0.3
                    bucket["score"] = round(normalized, 2)
                    scored.append(bucket)
            except Exception as e:
                logger.warning(f"Scoring failed for {bucket.get('id')}: {e}")

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _calc_topic_score(self, query: str, bucket: dict) -> float:
        meta = bucket.get("metadata", {})
        name_score = fuzz.partial_ratio(query, meta.get("name", "")) * 3
        domain_score = max(
            (fuzz.partial_ratio(query, d) for d in meta.get("domain", [])), default=0
        ) * 2.5
        tag_score = max(
            (fuzz.partial_ratio(query, tag) for tag in meta.get("tags", [])), default=0
        ) * 2
        content_score = fuzz.partial_ratio(query, bucket.get("content", "")[:1000]) * self.content_weight
        return (name_score + domain_score + tag_score + content_score) / (
            100 * (3 + 2.5 + 2 + self.content_weight)
        )

    def _calc_emotion_score(self, q_valence, q_arousal, meta) -> float:
        if q_valence is None or q_arousal is None:
            return 0.5
        try:
            b_valence = float(meta.get("valence", 0.5))
            b_arousal = float(meta.get("arousal", 0.3))
        except Exception:
            return 0.5
        dist = math.sqrt((q_valence - b_valence) ** 2 + (q_arousal - b_arousal) ** 2)
        return max(0.0, 1.0 - dist / 1.414)

    def _calc_time_score(self, meta) -> float:
        last_active_str = meta.get("last_active", meta.get("created", ""))
        try:
            last_active = datetime.fromisoformat(str(last_active_str))
            days = max(0.0, (datetime.now() - last_active).total_seconds() / 86400)
        except Exception:
            days = 30
        return math.exp(-0.02 * days)

    # ---------------------------------------------------------
    # List all
    # ---------------------------------------------------------
    async def list_all(self, include_archive: bool = False) -> list:
        res = self.supabase.table("buckets").select("*").execute()
        rows = res.data or []
        buckets = [_row_to_bucket(r) for r in rows]
        if not include_archive:
            buckets = [b for b in buckets if b["metadata"].get("type") != "archived"]
        return buckets

    # ---------------------------------------------------------
    # Stats
    # ---------------------------------------------------------
    async def get_stats(self) -> dict:
        all_buckets = await self.list_all(include_archive=True)
        stats = {
            "permanent_count": 0,
            "dynamic_count": 0,
            "archive_count": 0,
            "feel_count": 0,
            "total_size_kb": 0.0,
            "domains": {},
        }
        for b in all_buckets:
            t = b["metadata"].get("type", "dynamic")
            stats["total_size_kb"] += len((b.get("content") or "").encode()) / 1024
            if t == "permanent":
                stats["permanent_count"] += 1
            elif t == "dynamic":
                stats["dynamic_count"] += 1
            elif t == "archived":
                stats["archive_count"] += 1
            elif t == "feel":
                stats["feel_count"] += 1
        return stats

    # ---------------------------------------------------------
    # Archive
    # ---------------------------------------------------------
    async def archive(self, bucket_id: str) -> bool:
        self.supabase.table("buckets").update({
            "bucket_type": "archived"
        }).eq("id", bucket_id).execute()
        logger.info(f"Archived bucket: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Stubs for file-based methods (not used in DB mode)
    # ---------------------------------------------------------
    def _find_bucket_file(self, bucket_id: str):
        return None  # Not applicable in DB mode

    def _move_bucket(self, *args, **kwargs):
        pass  # Not applicable in DB mode
