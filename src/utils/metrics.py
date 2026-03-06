"""
Observability & Metrics — logs query performance to .refinery/metrics/query_log.jsonl.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class QueryMetrics:
    """Tracks and logs query-level metrics for observability."""

    def __init__(self, log_path: str = ".refinery/metrics/query_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_time: Optional[float] = None
        self._data: Dict[str, Any] = {}

    def start(self, query: str, document_id: str) -> None:
        self._start_time = time.time()
        self._data = {
            "query": query,
            "document_id": document_id,
            "tools_used": [],
            "llm_tokens_used": 0,
            "sql_direct_answer": False,
            "cost_usd": 0.0,
        }

    def record_tool(self, tool_name: str) -> None:
        self._data.setdefault("tools_used", []).append(tool_name)

    def record_tokens(self, tokens: int, cost_usd: float = 0.0) -> None:
        self._data["llm_tokens_used"] = self._data.get("llm_tokens_used", 0) + tokens
        self._data["cost_usd"] = self._data.get("cost_usd", 0.0) + cost_usd

    def record_sql_direct(self) -> None:
        self._data["sql_direct_answer"] = True

    def record_retrieval_scores(self, scores: list[float]) -> None:
        if scores:
            self._data["top_retrieval_score"] = round(max(scores), 4)
            self._data["avg_retrieval_score"] = round(sum(scores) / len(scores), 4)
            self._data["retrieval_hit_count"] = len(scores)

    def finish_and_log(self) -> Dict[str, Any]:
        if self._start_time:
            self._data["query_latency_ms"] = round((time.time() - self._start_time) * 1000, 1)

        import datetime
        self._data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(self._data) + "\n")
        except Exception as e:
            logger.warning(f"Metrics log write failed: {e}")

        return self._data
