"""
排名引擎：Elo 评分算法 + 智能配对选择 + 分数归一化。

Elo 是国际象棋等领域广泛使用的成对比较排名标准算法，
适合「两选一」偏好打分场景。
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import config


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pair_key(a: str, b: str) -> tuple[str, str]:
    """无序配对键，用于去重。"""
    return tuple(sorted((a, b)))


class RankingEngine:
    """管理 Elo 分数、对比记录与配对逻辑。"""

    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or config.RANKING_DATA_FILE
        self._data: dict[str, Any] = self._empty_data()
        self._load()

    @staticmethod
    def _empty_data() -> dict[str, Any]:
        return {
            "ratings": {},
            "comparison_counts": {},
            "comparisons": [],
            "pair_history": [],
            "current_round": 0,
            "finished": False,
            "known_faces": [],
        }

    def _load(self) -> None:
        if not self.data_path.exists():
            self._save()
            return
        try:
            with self.data_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self._data.update(loaded)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[警告] 读取排名数据失败，将使用空数据: {exc}")
            self._data = self._empty_data()

    def _save(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with self.data_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def sync_faces(self, face_ids: list[str]) -> None:
        """
        与磁盘扫描结果同步：新增图片初始化 Elo，移除已删除图片的记录。
        """
        face_set = set(face_ids)
        self._data["known_faces"] = sorted(face_set)

        for fid in face_set:
            self._data["ratings"].setdefault(fid, config.ELO_INITIAL_RATING)
            self._data["comparison_counts"].setdefault(fid, 0)

        stale = [k for k in self._data["ratings"] if k not in face_set]
        for k in stale:
            self._data["ratings"].pop(k, None)
            self._data["comparison_counts"].pop(k, None)

        # 清理历史中已不存在的图片
        self._data["comparisons"] = [
            c
            for c in self._data["comparisons"]
            if c.get("winner") in face_set and c.get("loser") in face_set
        ]
        self._data["pair_history"] = [
            p for p in self._data["pair_history"] if p[0] in face_set and p[1] in face_set
        ]

        self._save()

    @property
    def finished(self) -> bool:
        return bool(self._data.get("finished"))

    @property
    def current_round(self) -> int:
        return int(self._data.get("current_round", 0))

    @property
    def max_iterations(self) -> int:
        return config.MAX_ITERATIONS

    def get_status(self) -> dict[str, Any]:
        faces = self._data.get("known_faces", [])
        return {
            "current_round": self.current_round,
            "max_iterations": self.max_iterations,
            "finished": self.finished,
            "total_faces": len(faces),
            "remaining": max(0, self.max_iterations - self.current_round),
        }

    def stop_early(self) -> None:
        """手动结束对比流程。"""
        self._data["finished"] = True
        self._save()

    def reset_all(self) -> None:
        """重置所有排名与对比记录，保留已知图片列表。"""
        faces = list(self._data.get("known_faces", []))
        self._data = self._empty_data()
        self.sync_faces(faces)

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Elo 期望胜率 P(A 胜 B)。"""
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def update_elo(self, winner_id: str, loser_id: str) -> tuple[float, float]:
        """根据一次对比结果更新双方 Elo，返回 (新胜者分, 新败者分)。"""
        ra = float(self._data["ratings"].get(winner_id, config.ELO_INITIAL_RATING))
        rb = float(self._data["ratings"].get(loser_id, config.ELO_INITIAL_RATING))

        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)

        k = config.ELO_K_FACTOR
        new_ra = ra + k * (1.0 - ea)
        new_rb = rb + k * (0.0 - eb)

        self._data["ratings"][winner_id] = new_ra
        self._data["ratings"][loser_id] = new_rb
        return new_ra, new_rb

    def _recent_pair_set(self) -> set[tuple[str, str]]:
        history = self._data.get("pair_history", [])
        size = config.RECENT_PAIR_HISTORY_SIZE
        recent = history[-size:] if size > 0 else []
        return {_pair_key(a, b) for a, b in recent}

    def _pair_weight(self, a: str, b: str) -> float:
        """
        配对权重：对比次数越少权重越高，促进各图片对比频次均匀。
        """
        ca = self._data["comparison_counts"].get(a, 0)
        cb = self._data["comparison_counts"].get(b, 0)
        w = config.PAIR_SELECTION_UNDERCOMPARED_WEIGHT
        return (1.0 / (ca + 1)) ** w * (1.0 / (cb + 1)) ** w

    def select_next_pair(self) -> tuple[str, str] | None:
        """
        智能选择下一组对比图片：
        1. 排除最近出现过的配对；
        2. 优先选择对比次数较少的图片组合。
        """
        faces = self._data.get("known_faces", [])
        if len(faces) < 2:
            return None
        if self.finished or self.current_round >= self.max_iterations:
            return None

        recent = self._recent_pair_set()
        candidates: list[tuple[str, str, float]] = []

        for a, b in combinations(sorted(faces), 2):
            key = _pair_key(a, b)
            if key in recent:
                continue
            weight = self._pair_weight(a, b)
            candidates.append((a, b, weight))

        # 若全部被 recent 过滤，放宽限制但仍优先低频图片
        if not candidates:
            for a, b in combinations(sorted(faces), 2):
                weight = self._pair_weight(a, b)
                candidates.append((a, b, weight))

        if not candidates:
            return None

        ids, weights = zip(*[(c[:2], c[2]) for c in candidates])
        chosen = random.choices(list(ids), weights=list(weights), k=1)[0]
        # 随机左右顺序，避免位置偏差
        if random.random() < 0.5:
            return chosen[0], chosen[1]
        return chosen[1], chosen[0]

    def record_vote(self, left_id: str, right_id: str, choice: str) -> dict[str, Any]:
        """
        记录用户选择并更新 Elo。
        choice: 'left' 或 'right'
        """
        if self.finished:
            return {"ok": False, "error": "对比已结束"}

        if self.current_round >= self.max_iterations:
            self._data["finished"] = True
            self._save()
            return {"ok": False, "error": "已达到最大轮次"}

        if choice not in ("left", "right"):
            return {"ok": False, "error": "无效选择"}

        winner = left_id if choice == "left" else right_id
        loser = right_id if choice == "left" else left_id

        if winner == loser:
            return {"ok": False, "error": "左右图片不能相同"}

        known = set(self._data.get("known_faces", []))
        if winner not in known or loser not in known:
            return {"ok": False, "error": "图片不存在或已被移除"}

        new_w, new_l = self.update_elo(winner, loser)

        self._data["comparison_counts"][winner] = (
            self._data["comparison_counts"].get(winner, 0) + 1
        )
        self._data["comparison_counts"][loser] = (
            self._data["comparison_counts"].get(loser, 0) + 1
        )

        self._data["current_round"] = self.current_round + 1

        record = {
            "round": self._data["current_round"],
            "winner": winner,
            "loser": loser,
            "choice": choice,
            "winner_rating": round(new_w, 2),
            "loser_rating": round(new_l, 2),
            "timestamp": _utc_now_iso(),
        }
        self._data["comparisons"].append(record)

        pair = list(_pair_key(left_id, right_id))
        self._data.setdefault("pair_history", []).append(pair)
        max_hist = config.RECENT_PAIR_HISTORY_SIZE * 2
        if len(self._data["pair_history"]) > max_hist:
            self._data["pair_history"] = self._data["pair_history"][-max_hist:]

        if self._data["current_round"] >= self.max_iterations:
            self._data["finished"] = True

        self._save()

        return {
            "ok": True,
            "round": self._data["current_round"],
            "finished": self.finished,
            "winner_rating": new_w,
            "loser_rating": new_l,
        }

    def compute_rankings(self) -> list[dict[str, Any]]:
        """
        按 Elo 分数降序排名，并将排名线性映射到 0–10 分（保留 1 位小数）。
        排名第 1 → 10.0，排名 N → 0.0（N=1 时唯一图片得 10.0）。
        """
        faces = self._data.get("known_faces", [])
        if not faces:
            return []

        items = [
            {
                "id": fid,
                "elo": float(self._data["ratings"].get(fid, config.ELO_INITIAL_RATING)),
                "comparisons": int(self._data["comparison_counts"].get(fid, 0)),
            }
            for fid in faces
        ]
        items.sort(key=lambda x: (-x["elo"], x["id"]))

        n = len(items)
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
            if n == 1:
                item["score"] = 10.0
            else:
                # 线性归一化：rank 1 → 10.0, rank N → 0.0
                raw = 10.0 * (n - rank) / (n - 1)
                item["score"] = round(raw, 1)

        return items

    def get_total_comparisons(self) -> int:
        return len(self._data.get("comparisons", []))

    def export_csv_rows(self) -> list[list[str]]:
        """生成 CSV 行（含表头）。"""
        rows = [["rank", "filename", "score", "elo", "comparisons"]]
        for item in self.compute_rankings():
            rows.append(
                [
                    str(item["rank"]),
                    item["id"],
                    f"{item['score']:.1f}",
                    f"{item['elo']:.2f}",
                    str(item["comparisons"]),
                ]
            )
        return rows
