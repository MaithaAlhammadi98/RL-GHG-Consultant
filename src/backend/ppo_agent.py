"""
PPO Agent for GHG Consultant System
Designed to be directly comparable with Q-learning RLAgent
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .state import state_key


@dataclass
class PPOConfig:
    """PPO hyperparameters"""
    gamma: float = 0.9                # Same as Q-learning
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 32
    hidden_dim: int = 128
    buffer_size: int = 256           # Rollout buffer size


class StateEncoder:
    """Encode dict states to vectors for neural network"""
    
    TOPIC_MAP = {"ghg": 0, "legal": 1, "fin": 2, "other": 3}
    LENGTH_MAP = {"short": 0, "medium": 1, "long": 2}
    SECTOR_MAP = {"unknown": 0, "energy": 1, "finance": 2, "tech": 3, "other": 4}
    SIZE_MAP = {"unknown": 0, "small": 1, "medium": 2, "large": 3}
    
    @classmethod
    def encode(cls, state: Dict[str, Any]) -> np.ndarray:
        """Convert state dict to feature vector"""
        features = []
        
        # One-hot encode categorical features
        topic = cls.TOPIC_MAP.get(state.get("topic", "other"), 3)
        features.extend([1.0 if i == topic else 0.0 for i in range(4)])
        
        length = cls.LENGTH_MAP.get(state.get("len", "medium"), 1)
        features.extend([1.0 if i == length else 0.0 for i in range(3)])
        
        sector = cls.SECTOR_MAP.get(state.get("sector", "unknown"), 0)
        features.extend([1.0 if i == sector else 0.0 for i in range(5)])
        
        size = cls.SIZE_MAP.get(state.get("size", "unknown"), 0)
        features.extend([1.0 if i == size else 0.0 for i in range(4)])
        
        # Month encoding (normalize to 0-1)
        month = state.get("month", "2025-01")
        try:
            year, mon = map(int, month.split("-"))
            month_val = (year - 2024) * 12 + mon
            features.append(month_val / 24.0)  # Normalize
        except:
            features.append(0.5)
        
        return np.array(features, dtype=np.float32)
    
    @classmethod
    def state_dim(cls) -> int:
        """Total state dimension"""
        return 4 + 3 + 5 + 4 + 1  # topic + length + sector + size + month = 17


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for discrete actions"""
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, n_actions)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            (action_logits, state_value)
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value


class RolloutBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, size: int, state_dim: int):
        self.size = size
        self.ptr = 0
        
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            value: float, log_prob: float, done: bool):
        """Add transition"""
        idx = self.ptr % self.size
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        
        self.ptr += 1
    
    def compute_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE advantages"""
        last_gae = 0.0
        
        for t in reversed(range(min(self.ptr, self.size))):
            if t == min(self.ptr, self.size) - 1:
                next_value = last_value
                next_non_terminal = 1.0
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        self.returns = self.advantages + self.values
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors"""
        actual_size = min(self.ptr, self.size)
        return {
            'states': torch.FloatTensor(self.states[:actual_size]),
            'actions': torch.LongTensor(self.actions[:actual_size]),
            'old_log_probs': torch.FloatTensor(self.log_probs[:actual_size]),
            'advantages': torch.FloatTensor(self.advantages[:actual_size]),
            'returns': torch.FloatTensor(self.returns[:actual_size])
        }
    
    def clear(self):
        """Reset buffer"""
        self.ptr = 0


class PPOAgent:
    """
    PPO Agent for GHG Consultant System
    Compatible with existing Q-learning RLAgent interface
    """
    
    def __init__(
        self,
        actions: List[str] | None = None,
        config: Optional[PPOConfig] = None,
        model_path: Path | None = None,
        verbose: bool = False,
        device: Optional[torch.device] = None
    ):
        # Actions (same as Q-learning)
        self.actions = actions or ["broad", "legal_only", "financial_only", "company_only"]
        self.n_actions = len(self.actions)
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.idx_to_action = {i: a for i, a in enumerate(self.actions)}
        
        # Config
        self.config = config or PPOConfig()
        self.verbose = verbose
        
        # Device
        self.device = device or torch.device("cpu")
        
        # State encoding
        self.state_dim = StateEncoder.state_dim()
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            size=self.config.buffer_size,
            state_dim=self.state_dim
        )
        
        # Model path
        src_dir = Path(__file__).resolve().parents[1]
        data_dir = src_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path or (data_dir / "ppo_model.pt")
        
        # Try to load existing model
        if self.model_path.exists():
            self.load()
        
        # Tracking
        self.training_mode = False
        self.current_episode_transitions = []
        
        if self.verbose:
            print(f" PPO Agent initialized:")
            print(f"  Actions: {self.actions}")
            print(f"  State dim: {self.state_dim}")
            print(f"  Hidden dim: {self.config.hidden_dim}")
            print(f"  Device: {self.device}")
    
    def select(self, state: Dict[str, Any], deterministic: bool = False) -> str:
        """
        Select action (compatible with Q-learning interface)
        
        Args:
            state: State dictionary
            deterministic: If True, select best action (no exploration)
        
        Returns:
            Action string
        """
        # Encode state
        state_vec = StateEncoder.encode(state)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            logits, value = self.network(state_t)
            dist = Categorical(logits=logits)
            
            if deterministic:
                action_idx = logits.argmax(dim=-1).item()
            else:
                action_idx = dist.sample().item()
            
            action = self.idx_to_action[action_idx]
            
            # Store for training
            if self.training_mode:
                log_prob = dist.log_prob(torch.tensor([action_idx])).item()
                self.current_episode_transitions.append({
                    'state': state_vec,
                    'action': action_idx,
                    'value': value.item(),
                    'log_prob': log_prob
                })
            
            if self.verbose:
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                print(f"[PPO Select] Action: {action}, Probs: {dict(zip(self.actions, probs))}")
            
            return action
    
    def update(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any] | None = None,
    ) -> None:
        """
        Update agent (compatible with Q-learning interface)
        
        This method collects transitions and triggers PPO update when buffer is full
        """
        # Add reward to last transition
        if self.current_episode_transitions:
            last_transition = self.current_episode_transitions[-1]
            
            # Add to buffer
            self.buffer.add(
                state=last_transition['state'],
                action=last_transition['action'],
                reward=reward,
                value=last_transition['value'],
                log_prob=last_transition['log_prob'],
                done=next_state is None
            )
            
            if self.verbose:
                print(f"[PPO Update] Action: {action}, Reward: {reward:.3f}")
            
            # If episode done, clear transitions
            if next_state is None:
                self.current_episode_transitions = []
            
            # Trigger PPO update if buffer is full
            if self.buffer.ptr >= self.config.buffer_size:
                self._ppo_update()
                self.buffer.clear()
    
    def _ppo_update(self):
        """Perform PPO update on collected experience"""
        # Compute advantages
        with torch.no_grad():
            # Use zero as last value (episode ended)
            last_value = 0.0
        
        self.buffer.compute_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        # Get data
        data = self.buffer.get()
        
        # Normalize advantages
        advantages = data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        for epoch in range(self.config.n_epochs):
            indices = torch.randperm(len(advantages))
            
            for start in range(0, len(advantages), self.config.batch_size):
                end = min(start + self.config.batch_size, len(advantages))
                batch_idx = indices[start:end]
                
                # Get batch
                batch_states = data['states'][batch_idx].to(self.device)
                batch_actions = data['actions'][batch_idx].to(self.device)
                batch_old_log_probs = data['old_log_probs'][batch_idx].to(self.device)
                batch_advantages = advantages[batch_idx].to(self.device)
                batch_returns = data['returns'][batch_idx].to(self.device)
                
                # Forward pass
                logits, values = self.network(batch_states)
                dist = Categorical(logits=logits)
                
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                
                # PPO clipped loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        
        if self.verbose:
            print(f"[PPO Update] Buffer full, performed {self.config.n_epochs} epochs of updates")
    
    def best_action(self, state: Dict[str, Any]) -> str:
        """Get best action (for evaluation)"""
        return self.select(state, deterministic=True)
    
    def get_action_probs(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Get action probabilities (for inspection)"""
        state_vec = StateEncoder.encode(state)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            logits, _ = self.network(state_t)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        
        return {action: float(prob) for action, prob in zip(self.actions, probs)}
    
    def start_episode(self):
        """Start training episode"""
        self.training_mode = True
        self.current_episode_transitions = []
    
    def end_episode(self):
        """End training episode"""
        self.training_mode = False
        self.current_episode_transitions = []
    
    def save(self, path: Optional[Path] = None):
        """Save model"""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'actions': self.actions
        }, save_path)
        
        if self.verbose:
            print(f" PPO model saved to {save_path}")
    
    def load(self, path: Optional[Path] = None):
        """Load model"""
        load_path = path or self.model_path
        
        if not load_path.exists():
            if self.verbose:
                print(f"️ No model found at {load_path}")
            return
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.verbose:
            print(f" PPO model loaded from {load_path}")
    
    def print_policy(self):
        """Print policy (like Q-table for Q-learning)"""
        print(" PPO Policy Overview:")
        
        # Sample some states
        sample_states = [
            {"topic": "ghg", "len": "short", "sector": "unknown", "size": "unknown", "month": "2025-10"},
            {"topic": "legal", "len": "short", "sector": "unknown", "size": "unknown", "month": "2025-10"},
            {"topic": "fin", "len": "short", "sector": "unknown", "size": "unknown", "month": "2025-10"},
        ]
        
        for state in sample_states:
            probs = self.get_action_probs(state)
            print(f"\nState: {state}")
            for action, prob in sorted(probs.items(), key=lambda x: -x[1]):
                print(f"  {action}: {prob:.3f}")


# Comparison utilities
class AgentComparison:
    """Compare Q-learning and PPO agents"""
    
    @staticmethod
    def compare_agents(
        q_agent,
        ppo_agent,
        test_states: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare action selection between agents"""
        comparison = {
            'agreements': 0,
            'disagreements': 0,
            'details': []
        }
        
        for state in test_states:
            q_action = q_agent.best_action(state)
            ppo_action = ppo_agent.best_action(state)
            
            agree = q_action == ppo_action
            if agree:
                comparison['agreements'] += 1
            else:
                comparison['disagreements'] += 1
            
            # Get Q-values and PPO probs
            q_values = q_agent.q_for(state)
            ppo_probs = ppo_agent.get_action_probs(state)
            
            comparison['details'].append({
                'state': state,
                'q_action': q_action,
                'ppo_action': ppo_action,
                'agree': agree,
                'q_values': q_values,
                'ppo_probs': ppo_probs
            })
        
        comparison['agreement_rate'] = comparison['agreements'] / len(test_states)
        
        return comparison
    
    @staticmethod
    def print_comparison(comparison: Dict[str, Any]):
        """Print comparison results"""
        print(" Q-Learning vs PPO Comparison")
        print("="*60)
        print(f"Agreement Rate: {comparison['agreement_rate']:.1%}")
        print(f"Agreements: {comparison['agreements']}")
        print(f"Disagreements: {comparison['disagreements']}")
        print()
        
        print("Detailed Comparisons:")
        for i, detail in enumerate(comparison['details'][:5], 1):  # Show first 5
            print(f"\n{i}. State: {detail['state']['topic']}/{detail['state']['len']}")
            print(f"   Q-Learning  {detail['q_action']} (Q-values: {detail['q_values']})")
            print(f"   PPO  {detail['ppo_action']} (Probs: {detail['ppo_probs']})")
            print(f"   {' AGREE' if detail['agree'] else ' DISAGREE'}")

