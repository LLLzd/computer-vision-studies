"""
排名引擎：TrueSkill 双参数(μ/σ) + 智能配对 + 收敛判定。

核心点：
1) 每张图片维护 μ(能力均值) 与 σ(不确定性方差的标准差)；
2) 每次偏好比较按 TrueSkill 1v1 更新；
3) 有效得分使用 μ - 3σ，兼顾水平与置信度；
4) 引入新样本公平校准、抗刷分抑制、收敛停止机制。
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
    """管理 TrueSkill 分数、对比记录与配对逻辑。"""

    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or config.RANKING_DATA_FILE
        self._data: dict[str, Any] = self._empty_data()
        self._load()

    @staticmethod
    def _empty_data() -> dict[str, Any]:
        return {
            "trueskill": {},
            "comparison_counts": {},
            "comparisons": [],
            "pair_history": [],
            "recent_faces": [],
            "recent_score_deltas": [],
            "current_round": 0,
            "finished": False,
            "converged": False,
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
            self._migrate_legacy_schema()
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[警告] 读取排名数据失败，将使用空数据: {exc}")
            self._data = self._empty_data()

    def _migrate_legacy_schema(self) -> None:
        """
        兼容旧数据结构：
        - 旧版可能只有 ratings(Elo)；
        - 新版统一使用 trueskill: {id: {mu, sigma}}。
        """
        trueskill_map = self._data.get("trueskill")
        if not isinstance(trueskill_map, dict):
            trueskill_map = {}
            self._data["trueskill"] = trueskill_map

        # 将旧 Elo 结构平滑迁移为统一初始 TrueSkill（避免旧字段导致崩溃）
        ratings = self._data.get("ratings", {})
        if isinstance(ratings, dict) and ratings:
            for fid in ratings:
                trueskill_map.setdefault(
                    fid,
                    {
                        "mu": float(config.TRUESKILL_MU),
                        "sigma": float(config.TRUESKILL_SIGMA),
                    },
                )
            self._data.pop("ratings", None)

        # 字段兜底
        self._data.setdefault("comparison_counts", {})
        self._data.setdefault("comparisons", [])
        self._data.setdefault("pair_history", [])
        self._data.setdefault("recent_faces", [])
        self._data.setdefault("recent_score_deltas", [])
        self._data.setdefault("current_round", 0)
        self._data.setdefault("finished", False)
        self._data.setdefault("converged", False)
        self._data.setdefault("known_faces", [])

    def _save(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with self.data_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def sync_faces(self, face_ids: list[str]) -> None:
        """
        与磁盘扫描结果同步：
        - 新增图片初始化为统一 TrueSkill 起点；
        - 移除已删除图片的记录；
        - 清理历史中无效 ID。
        """
        face_set = set(face_ids)
        self._data["known_faces"] = sorted(face_set)

        ts_map = self._data.setdefault("trueskill", {})
        for fid in face_set:
            ts_map.setdefault(
                fid,
                {
                    "mu": float(config.TRUESKILL_MU),
                    "sigma": float(config.TRUESKILL_SIGMA),
                },
            )
            self._data["comparison_counts"].setdefault(fid, 0)

        stale = [k for k in ts_map if k not in face_set]
        for k in stale:
            ts_map.pop(k, None)
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
        self._data["recent_faces"] = [
            f for f in self._data.get("recent_faces", []) if f in face_set
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
        return int(config.MAX_ITERATIONS)

    @property
    def unlimited_iterations(self) -> bool:
        return self.max_iterations <= 0

    def get_status(self) -> dict[str, Any]:
        faces = self._data.get("known_faces", [])
        converged = bool(self._data.get("converged", False))
        return {
            "current_round": self.current_round,
            "max_iterations": self.max_iterations,
            "unlimited_iterations": self.unlimited_iterations,
            "finished": self.finished,
            "converged": converged,
            "total_faces": len(faces),
            "remaining": None
            if self.unlimited_iterations
            else max(0, self.max_iterations - self.current_round),
        }

    def ensure_comparison_active(self) -> None:
        """确保对比可持续进行（兼容旧数据中 finished=true 的情况）。"""
        if self._data.get("finished"):
            self._data["finished"] = False
            self._save()

    def reset_all(self) -> None:
        """重置所有排名与对比记录，保留已知图片列表。"""
        faces = list(self._data.get("known_faces", []))
        self._data = self._empty_data()
        self.sync_faces(faces)

    @staticmethod
    def _normal_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _v_win(self, t: float) -> float:
        denom = max(1e-9, self._normal_cdf(t))
        return self._normal_pdf(t) / denom

    def _w_win(self, t: float, v: float) -> float:
        return v * (v + t)

    def _get_rating(self, face_id: str) -> tuple[float, float]:
        info = self._data.get("trueskill", {}).get(face_id, {})
        mu = float(info.get("mu", config.TRUESKILL_MU))
        sigma = float(info.get("sigma", config.TRUESKILL_SIGMA))
        sigma = min(max(sigma, config.TRUESKILL_SIGMA_MIN), config.TRUESKILL_SIGMA_MAX)
        return mu, sigma

    def _set_rating(self, face_id: str, mu: float, sigma: float) -> None:
        sigma = min(max(sigma, config.TRUESKILL_SIGMA_MIN), config.TRUESKILL_SIGMA_MAX)
        self._data.setdefault("trueskill", {})[face_id] = {"mu": float(mu), "sigma": float(sigma)}

    def _effective_score(self, mu: float, sigma: float) -> float:
        return mu - config.EFFECTIVE_SCORE_SIGMA_FACTOR * sigma

    def _comparison_impact(self, a: str, b: str) -> float:
        """
        动态校准更新强度：
        - 低频样本提升：加快新样本公平收敛；
        - 分差抑制：降低“碾压局”对分数的单次冲击，避免刷分。
        """
        ca = int(self._data["comparison_counts"].get(a, 0))
        cb = int(self._data["comparison_counts"].get(b, 0))
        low = min(ca, cb)
        target = max(1, int(config.LOW_FREQUENCY_TARGET))
        low_ratio = max(0.0, float(target - low) / float(target))
        low_boost = 1.0 + low_ratio * float(config.LOW_FREQUENCY_UPDATE_BOOST)

        mu_a, sigma_a = self._get_rating(a)
        mu_b, sigma_b = self._get_rating(b)
        gap = abs(self._effective_score(mu_a, sigma_a) - self._effective_score(mu_b, sigma_b))
        crush = 1.0 / (
            1.0 + (gap / max(1e-6, float(config.CRUSH_GAP_SCALE))) * float(config.CRUSH_PENALTY_STRENGTH)
        )

        impact = low_boost * crush
        return min(max(impact, config.UPDATE_IMPACT_MIN), config.UPDATE_IMPACT_MAX)

    def update_trueskill(self, winner_id: str, loser_id: str) -> dict[str, float]:
        """
        1v1 TrueSkill 更新（胜负场景）：
        - 使用 μ/σ 双参数迭代；
        - 有效得分基于 μ-3σ；
        - 叠加动态权重，兼顾新样本公平与抗刷分稳定性。
        """
        mu_w, sigma_w = self._get_rating(winner_id)
        mu_l, sigma_l = self._get_rating(loser_id)

        # 动态波动项：避免 σ 过快塌缩导致模型僵化
        sigma_w_star = math.sqrt(sigma_w * sigma_w + config.TRUESKILL_TAU * config.TRUESKILL_TAU)
        sigma_l_star = math.sqrt(sigma_l * sigma_l + config.TRUESKILL_TAU * config.TRUESKILL_TAU)

        c = math.sqrt(
            2.0 * config.TRUESKILL_BETA * config.TRUESKILL_BETA
            + sigma_w_star * sigma_w_star
            + sigma_l_star * sigma_l_star
        )
        c = max(c, 1e-9)

        t = (mu_w - mu_l) / c
        v = self._v_win(t)
        w = self._w_win(t, v)

        impact = self._comparison_impact(winner_id, loser_id)
        v *= impact
        w *= impact

        mu_w_new = mu_w + (sigma_w_star * sigma_w_star / c) * v
        mu_l_new = mu_l - (sigma_l_star * sigma_l_star / c) * v

        # 方差更新，做数值保护防止出现负数
        sigma_w_var = sigma_w_star * sigma_w_star * (
            1.0 - (sigma_w_star * sigma_w_star / (c * c)) * w
        )
        sigma_l_var = sigma_l_star * sigma_l_star * (
            1.0 - (sigma_l_star * sigma_l_star / (c * c)) * w
        )
        sigma_w_var = max(sigma_w_var, config.TRUESKILL_SIGMA_MIN * config.TRUESKILL_SIGMA_MIN)
        sigma_l_var = max(sigma_l_var, config.TRUESKILL_SIGMA_MIN * config.TRUESKILL_SIGMA_MIN)

        sigma_w_new = math.sqrt(sigma_w_var)
        sigma_l_new = math.sqrt(sigma_l_var)

        self._set_rating(winner_id, mu_w_new, sigma_w_new)
        self._set_rating(loser_id, mu_l_new, sigma_l_new)

        return {
            "winner_mu": mu_w_new,
            "winner_sigma": sigma_w_new,
            "loser_mu": mu_l_new,
            "loser_sigma": sigma_l_new,
            "winner_effective": self._effective_score(mu_w_new, sigma_w_new),
            "loser_effective": self._effective_score(mu_l_new, sigma_l_new),
        }

    def _recent_pair_set(self) -> set[tuple[str, str]]:
        history = self._data.get("pair_history", [])
        size = config.RECENT_PAIR_HISTORY_SIZE
        recent = history[-size:] if size > 0 else []
        return {_pair_key(a, b) for a, b in recent}

    def _pair_weight(self, a: str, b: str) -> float:
        """
        配对权重（多因素）：
        1) 低频次优先，保证每张图参与次数均匀；
        2) 高不确定性优先，加快收敛；
        3) 分数相近优先，提高信息增益；
        4) 短期冷却，避免同一图片连续出现。
        """
        ca = self._data["comparison_counts"].get(a, 0)
        cb = self._data["comparison_counts"].get(b, 0)
        under_w = config.PAIR_SELECTION_UNDERCOMPARED_WEIGHT
        under = (1.0 / (ca + 1)) ** under_w * (1.0 / (cb + 1)) ** under_w

        _, sigma_a = self._get_rating(a)
        _, sigma_b = self._get_rating(b)
        sigma_ref = max(1e-6, float(config.TRUESKILL_SIGMA))
        uncertainty = ((sigma_a + sigma_b) / (2.0 * sigma_ref)) ** float(
            config.PAIR_SELECTION_UNCERTAINTY_WEIGHT
        )

        mu_a, _ = self._get_rating(a)
        mu_b, _ = self._get_rating(b)
        eff_a = self._effective_score(mu_a, sigma_a)
        eff_b = self._effective_score(mu_b, sigma_b)
        gap = abs(eff_a - eff_b)
        info_gain = (
            1.0
            / (1.0 + gap / max(1e-6, float(config.PAIR_INFO_GAIN_GAP_SCALE)))
        ) ** float(config.PAIR_INFO_GAIN_WEIGHT)

        recent_faces = self._data.get("recent_faces", [])[-config.RECENT_FACE_COOLDOWN_SIZE :]
        cooldown_penalty = 1.0
        if a in recent_faces:
            cooldown_penalty *= float(config.RECENT_FACE_COOLDOWN_PENALTY)
        if b in recent_faces:
            cooldown_penalty *= float(config.RECENT_FACE_COOLDOWN_PENALTY)

        return max(1e-9, under * uncertainty * info_gain * cooldown_penalty)

    def select_next_pair(self) -> tuple[str, str] | None:
        """
        智能选择下一组对比图片：
        1. 排除最近出现过的配对；
        2. 优先选择对比次数较少的图片组合。
        """
        faces = self._data.get("known_faces", [])
        if len(faces) < 2:
            return None
        self.ensure_comparison_active()
        if (not self.unlimited_iterations) and self.current_round >= self.max_iterations:
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

    def _append_recent_face(self, face_id: str) -> None:
        history = self._data.setdefault("recent_faces", [])
        history.append(face_id)
        max_keep = max(2, int(config.RECENT_FACE_COOLDOWN_SIZE) * 3)
        if len(history) > max_keep:
            self._data["recent_faces"] = history[-max_keep:]

    def _append_delta(self, value: float) -> None:
        deltas = self._data.setdefault("recent_score_deltas", [])
        deltas.append(float(value))
        max_keep = max(10, int(config.CONVERGENCE_WINDOW) * 4)
        if len(deltas) > max_keep:
            self._data["recent_score_deltas"] = deltas[-max_keep:]

    def _is_converged(self) -> bool:
        if self.current_round < int(config.CONVERGENCE_MIN_ROUNDS):
            return False

        faces = self._data.get("known_faces", [])
        if len(faces) < 2:
            return False

        min_comp = min(int(self._data["comparison_counts"].get(fid, 0)) for fid in faces)
        if min_comp < int(config.CONVERGENCE_MIN_COMPARISONS_PER_FACE):
            return False

        window = max(1, int(config.CONVERGENCE_WINDOW))
        deltas = self._data.get("recent_score_deltas", [])
        if len(deltas) < window:
            return False

        recent_avg = sum(deltas[-window:]) / float(window)
        return recent_avg <= float(config.CONVERGENCE_DELTA_THRESHOLD)

    def record_vote(self, left_id: str, right_id: str, choice: str) -> dict[str, Any]:
        """
        记录用户选择并更新 TrueSkill。
        choice: 'left' 或 'right'
        """
        self.ensure_comparison_active()

        if (not self.unlimited_iterations) and self.current_round >= self.max_iterations:
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

        old_w_mu, old_w_sigma = self._get_rating(winner)
        old_l_mu, old_l_sigma = self._get_rating(loser)
        old_w_eff = self._effective_score(old_w_mu, old_w_sigma)
        old_l_eff = self._effective_score(old_l_mu, old_l_sigma)
        updated = self.update_trueskill(winner, loser)

        self._data["comparison_counts"][winner] = (
            self._data["comparison_counts"].get(winner, 0) + 1
        )
        self._data["comparison_counts"][loser] = (
            self._data["comparison_counts"].get(loser, 0) + 1
        )

        self._data["current_round"] = self.current_round + 1

        # 用有效得分变化量做收敛判定
        delta = (
            abs(updated["winner_effective"] - old_w_eff)
            + abs(updated["loser_effective"] - old_l_eff)
        ) / 2.0
        self._append_delta(delta)

        record = {
            "round": self._data["current_round"],
            "winner": winner,
            "loser": loser,
            "choice": choice,
            "winner_mu": round(updated["winner_mu"], 4),
            "winner_sigma": round(updated["winner_sigma"], 4),
            "winner_effective": round(updated["winner_effective"], 4),
            "loser_mu": round(updated["loser_mu"], 4),
            "loser_sigma": round(updated["loser_sigma"], 4),
            "loser_effective": round(updated["loser_effective"], 4),
            "delta_effective": round(delta, 6),
            "timestamp": _utc_now_iso(),
        }
        self._data["comparisons"].append(record)

        pair = list(_pair_key(left_id, right_id))
        self._data.setdefault("pair_history", []).append(pair)
        max_hist = config.RECENT_PAIR_HISTORY_SIZE * 2
        if len(self._data["pair_history"]) > max_hist:
            self._data["pair_history"] = self._data["pair_history"][-max_hist:]

        self._append_recent_face(left_id)
        self._append_recent_face(right_id)

        converged_now = self._is_converged()
        if converged_now:
            self._data["converged"] = True
        if (not self.unlimited_iterations) and self._data["current_round"] >= self.max_iterations:
            self._data["finished"] = True

        self._save()

        return {
            "ok": True,
            "round": self._data["current_round"],
            "finished": self.finished,
            "winner_mu": updated["winner_mu"],
            "winner_sigma": updated["winner_sigma"],
            "loser_mu": updated["loser_mu"],
            "loser_sigma": updated["loser_sigma"],
        }

    def compute_rankings(self) -> list[dict[str, Any]]:
        """
        按有效得分(μ-3σ)降序排名，并线性归一化到 0–10（保留 1 位小数）。
        若当前所有样本有效得分相同，则统一展示为 5.0（初始公平基线）。
        """
        faces = self._data.get("known_faces", [])
        if not faces:
            return []

        items = [
            {
                "id": fid,
                "mu": float(self._get_rating(fid)[0]),
                "sigma": float(self._get_rating(fid)[1]),
                "comparisons": int(self._data["comparison_counts"].get(fid, 0)),
            }
            for fid in faces
        ]
        for item in items:
            item["effective_score"] = self._effective_score(item["mu"], item["sigma"])

        items.sort(key=lambda x: (-x["effective_score"], -x["mu"], x["id"]))

        max_eff = max(item["effective_score"] for item in items)
        min_eff = min(item["effective_score"] for item in items)
        score_span = max_eff - min_eff

        n = len(items)
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
            if n == 1 or abs(score_span) < 1e-9:
                item["score"] = 5.0
            else:
                raw = 10.0 * (item["effective_score"] - min_eff) / score_span
                item["score"] = round(raw, 1)

        return items

    def get_total_comparisons(self) -> int:
        return len(self._data.get("comparisons", []))

    def export_csv_rows(self) -> list[list[str]]:
        """生成 CSV 行（含表头）。"""
        rows = [["rank", "filename", "score_0_10", "effective_score", "mu", "sigma", "comparisons"]]
        for item in self.compute_rankings():
            rows.append(
                [
                    str(item["rank"]),
                    item["id"],
                    f"{item['score']:.1f}",
                    f"{item['effective_score']:.4f}",
                    f"{item['mu']:.4f}",
                    f"{item['sigma']:.4f}",
                    str(item["comparisons"]),
                ]
            )
        return rows
