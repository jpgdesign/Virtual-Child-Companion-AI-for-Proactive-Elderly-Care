from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def push(
        self,
        state: Sequence[float],
        action: Sequence[float] | int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> None:
        action_idx = int(np.argmax(action)) if isinstance(action, (list, tuple, np.ndarray)) else int(action)
        transition = (
            np.asarray(state, dtype=np.float32),
            action_idx,
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[tuple[np.ndarray, int, float, np.ndarray, bool]]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class QLearningHyperparameters:
    state_dim: int
    action_dim: int
    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    memory_capacity: int
    batch_size: int


class TabularQLearningAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        memory_capacity: int = 50_000,
        batch_size: int = 64,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_capacity)
        self.q_table = np.zeros((2 ** state_dim, action_dim), dtype=np.float32)
        self.training_losses: List[float] = []
        self.training_rewards: List[float] = []
        self.update_count = 0
        self.total_steps = 0
        self.hyperparameters = QLearningHyperparameters(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            memory_capacity=memory_capacity,
            batch_size=batch_size,
        )

    def _state_to_index(self, state: Sequence[float]) -> int:
        bits = [1 if float(value) > 0 else 0 for value in state[: self.state_dim]]
        index = 0
        for offset, bit in enumerate(bits):
            index += bit << (self.state_dim - offset - 1)
        return index

    def get_q_values(self, state: Sequence[float]) -> np.ndarray:
        return self.q_table[self._state_to_index(state)].copy()

    def select_action(self, state: Sequence[float], training: bool = True) -> np.ndarray:
        if training and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
        else:
            action_idx = int(np.argmax(self.get_q_values(state)))

        action = np.zeros(self.action_dim, dtype=np.float32)
        action[action_idx] = 1.0
        return action

    def train(self, batch_size: int | None = None) -> float:
        batch_size = batch_size or self.batch_size
        if len(self.memory) < batch_size:
            return 0.0

        transitions = self.memory.sample(batch_size)
        td_errors = []
        rewards = []

        for state, action_idx, reward, next_state, done in transitions:
            state_idx = self._state_to_index(state)
            next_state_idx = self._state_to_index(next_state)

            current_q = self.q_table[state_idx, action_idx]
            next_q = 0.0 if done else float(np.max(self.q_table[next_state_idx]))
            target_q = reward + self.gamma * next_q
            td_error = target_q - current_q
            self.q_table[state_idx, action_idx] += self.alpha * td_error

            td_errors.append(abs(float(td_error)))
            rewards.append(float(reward))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.update_count += 1
        self.total_steps += batch_size
        mean_td_error = float(np.mean(td_errors))
        self.training_losses.append(mean_td_error)
        self.training_rewards.append(float(np.mean(rewards)))
        return mean_td_error

    def save_model(self, model_path: str) -> None:
        payload = {
            "algorithm": "q_learning",
            "q_table": self.q_table.tolist(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "total_steps": self.total_steps,
            "training_rewards": self.training_rewards,
            "training_losses": self.training_losses,
            "hyperparameters": asdict(self.hyperparameters),
        }
        Path(model_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_model(self, model_path: str) -> None:
        payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.q_table = np.asarray(payload["q_table"], dtype=np.float32)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.update_count = payload.get("update_count", self.update_count)
        self.total_steps = payload.get("total_steps", self.total_steps)
        self.training_rewards = payload.get("training_rewards", [])
        self.training_losses = payload.get("training_losses", [])
