from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json, random, threading

# keep this as a relative import since rl_agent.py is inside backend/
from .state import state_key

DEFAULT_ACTIONS: List[str] = ["broad", "legal_only", "financial_only", "company_only"]


class RLAgent:
    def __init__(
        self,
        actions: List[str] | None = None,
        epsilon: float = 0.2,
        alpha: float = 0.3,
        gamma: float = 0.9,
        q_path: Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.actions = actions or DEFAULT_ACTIONS
        self.epsilon = float(epsilon)
        self.alpha   = float(alpha)
        self.gamma   = float(gamma)
        self.verbose = verbose

        # data dir: <repo>/src/data/q_table.json
        src_dir  = Path(__file__).resolve().parents[1]
        data_dir = src_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.q_path = q_path or (data_dir / "q_table.json")

        self._lock = threading.Lock()
        self._Q: Dict[str, Dict[str, float]] = self._load()

    # ---------- persistence ----------
    def _load(self) -> Dict[str, Dict[str, float]]:
        if self.q_path.exists():
            try:
                with self.q_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        try:
            tmp = self.q_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._Q, f, indent=2, ensure_ascii=False)
            
            # Handle Windows file lock issues
            try:
                tmp.replace(self.q_path)
            except PermissionError:
                # If file is locked, try direct write
                with self.q_path.open("w", encoding="utf-8") as f:
                    json.dump(self._Q, f, indent=2, ensure_ascii=False)
                # Clean up temp file
                if tmp.exists():
                    tmp.unlink()
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Failed to save Q-table: {e}")

    # ---------- tie-breaker + helpers ----------
    def _argmax_random(self, q: Dict[str, float]) -> str:
        """Return a random action among those tied for the max Q."""
        if not q:  # safety
            return random.choice(self.actions)
        maxv = max(q.values())
        best = [a for a, v in q.items() if v == maxv]
        return random.choice(best)

    def _ensure_state(self, s: str) -> Dict[str, float]:
        if s not in self._Q:
            self._Q[s] = {a: 0.0 for a in self.actions}
        return self._Q[s]

    # ---------- policy ----------
    def _epsilon_greedy(self, s: str) -> str:
        # explore
        if random.random() < self.epsilon:
            a = random.choice(self.actions)
            if self.verbose:
                print(f"[Explore] Random action: {a}")
            return a
        # exploit (with random tie-break on equal Qs)
        q = self._ensure_state(s)
        a = self._argmax_random(q)
        if self.verbose:
            print(f"[Exploit] Best action: {a} with Q={q[a]:.3f}")
        return a

    # ---------- public API ----------
    def select(self, state: Dict[str, Any]) -> str:
        s = state_key(state)
        with self._lock:
            return self._epsilon_greedy(s)

    def update(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any] | None = None,
    ) -> None:
        s = state_key(state)
        with self._lock:
            q = self._ensure_state(s)
            old = q.get(action, 0.0)

            best_next = 0.0
            if next_state is not None:
                ns = state_key(next_state)
                best_next = max(self._ensure_state(ns).values(), default=0.0)

            target = reward + self.gamma * best_next
            q[action] = old + self.alpha * (target - old)

            if self.verbose:
                print(
                    f"[Update] State={s}, Action={action}, OldQ={old:.3f}, "
                    f"Reward={reward:.3f}, Target={target:.3f}, NewQ={q[action]:.3f}"
                )

            self._save()

    # ---------- inspection ----------
    def q_for(self, state: Dict[str, Any]) -> Dict[str, float]:
        s = state_key(state)
        return self._ensure_state(s)

    def best_action(self, state: Dict[str, Any]) -> str:
        s = state_key(state)
        q = self._ensure_state(s)
        return self._argmax_random(q)  # <- consistent tie-breaking

    # ---------- misc ----------
    def decay_epsilon(self, factor: float = 0.99, min_eps: float = 0.01) -> float:
        self.epsilon = max(min_eps, self.epsilon * factor)
        if self.verbose:
            print(f"[eps] -> {self.epsilon:.4f}")
        return self.epsilon

    def set_verbose(self, v: bool = True) -> None:
        self.verbose = v
    def print_q_table(self) -> None:
        for state, actions in self._Q.items():
            print(f"State: {state}")
            for action, q_value in actions.items():
                print(f"  Action: {action}, Q-value: {q_value:.3f}")
            print()