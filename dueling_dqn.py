from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: Sequence[float],
        action: Sequence[float] | int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> None:
        action_idx = int(np.argmax(action)) if isinstance(action, (list, tuple, np.ndarray)) else int(action)
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                action_idx,
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_head(features)
        advantages = self.advantage_head(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


@dataclass
class DQNHyperparameters:
    state_dim: int
    action_dim: int
    learning_rate: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    target_update: int
    memory_capacity: int
    batch_size: int
    weight_decay: float
    use_weighted_loss: bool
    positive_weight: float
    negative_weight: float


class DuelingDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        target_update: int = 100,
        memory_capacity: int = 50_000,
        batch_size: int = 64,
        weight_decay: float = 0.0,
        use_weighted_loss: bool = False,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.use_weighted_loss = use_weighted_loss
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.loss_fn = nn.MSELoss(reduction="none" if use_weighted_loss else "mean")
        self.memory = ReplayBuffer(memory_capacity)
        self.update_count = 0
        self.total_steps = 0
        self.training_rewards: List[float] = []
        self.training_losses: List[float] = []
        self.hyperparameters = DQNHyperparameters(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            target_update=target_update,
            memory_capacity=memory_capacity,
            batch_size=batch_size,
            weight_decay=weight_decay,
            use_weighted_loss=use_weighted_loss,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
        )

    def get_q_values(self, state: Sequence[float]) -> np.ndarray:
        state_tensor = torch.tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
        return q_values

    def select_action(self, state: Sequence[float], training: bool = True) -> np.ndarray:
        if training and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
        else:
            q_values = self.get_q_values(state)
            action_idx = int(np.argmax(q_values))

        action = np.zeros(self.action_dim, dtype=np.float32)
        action[action_idx] = 1.0
        return action

    def train(self, batch_size: int | None = None) -> float:
        batch_size = batch_size or self.batch_size
        if len(self.memory) < batch_size:
            return 0.0

        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_tensor = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.q_network(states_tensor).gather(1, actions_tensor).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states_tensor).max(dim=1).values
            target_q = rewards_tensor + self.gamma * next_q * (1.0 - dones_tensor)

        loss = self.loss_fn(current_q, target_q)
        if self.use_weighted_loss:
            weights = torch.where(rewards_tensor > 0, self.positive_weight, self.negative_weight)
            loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.update_count += 1
        self.total_steps += batch_size
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        loss_value = float(loss.item())
        self.training_losses.append(loss_value)
        self.training_rewards.append(float(np.mean(rewards)))
        return loss_value

    def save_model(self, model_path: str) -> None:
        torch.save(
            {
                "algorithm": "dqn",
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
                "total_steps": self.total_steps,
                "training_rewards": self.training_rewards,
                "training_losses": self.training_losses,
                "hyperparameters": self.hyperparameters.__dict__,
            },
            model_path,
        )

    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.update_count = checkpoint.get("update_count", self.update_count)
        self.total_steps = checkpoint.get("total_steps", self.total_steps)
        self.training_rewards = checkpoint.get("training_rewards", [])
        self.training_losses = checkpoint.get("training_losses", [])
