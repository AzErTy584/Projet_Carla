import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Confisuratioon du devices pour forcer l'utilisation des CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classe pour modèle PPO
class PPOActorCritic(nn.Module):
    def __init__(
        self,
        lidar_dim=32,
        action_dim=3,          # Rapel steering, throttle, brake
        hidden_dim=256
    ):
        super().__init__()

        self.state_dim = lidar_dim + 7  # collision(1) + speed(1) + lane_offset(1) + lane_angle(1) + (goal_direction(2) + goal_distance(1) = pas utilisé)

        # Backbone partagée pour extraire les features de l'état
        self.backbone = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Réseau acteur
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        
        # Initialisation équilibrée pour éviter les biais
        nn.init.xavier_uniform_(self.actor_mean.weight, gain=0.1)
        nn.init.zeros_(self.actor_mean.bias)
        
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * 0.5)

        # Réseau critique
        self.critic = nn.Linear(hidden_dim, 1)

    # Forward pass pour obtenir les actions et la valeur d'état
    def forward(self, state_tensor):

        features = self.backbone(state_tensor)

        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)

        value = self.critic(features).squeeze(-1)

        return mean, std, value

    # Action sampling pour l'interaction avec l'environnement
    def act(self, state_tensor):

        mean, std, value = self.forward(state_tensor)
        dist = Normal(mean, std)

        raw_action = dist.sample()
        action = torch.sigmoid(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        return action, log_prob, value

    # Evaluation pour le calcul des log-probs et de l'entropie pendant l'update PPO
    def evaluate(self, state_tensor, action):

        mean, std, value = self.forward(state_tensor)
        dist = Normal(mean, std)

        # Inverse tanh pour retrouver raw_action depuis l'action tanh
        raw_action = torch.atanh(torch.clamp(action, -0.99, 0.99))
        
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

# Classe pour stocker les transitions de l'épisode
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

# Fonction pour calculer les avantages et les retours à partir des transitions stockées
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

# Fonction de mise à jour PPO
def ppo_update(
    model,
    optimizer,
    buffer,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.05,
    epochs=10
):
    advantages, returns = compute_gae(
        buffer.rewards,
        buffer.values,
        buffer.dones
    )

    advantages = torch.tensor(advantages, device=device)
    returns = torch.tensor(returns, device=device)

    states = torch.stack(buffer.states)
    actions = torch.stack(buffer.actions)
    old_log_probs = torch.tensor(buffer.log_probs, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        log_probs, entropy, values = model.evaluate(states, actions)

        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, returns)

        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Sauvegarde
def save_model(model, optimizer, episode, path="V0\model_checkpoint_PPO.pth"):
    try:
        torch.save({
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print(f"[MODEL] Sauvegardé épisode {episode} -> {path}")
    except Exception as e:
        print("[MODEL] Erreur sauvegarde:", e)

# Chargement modèle
def load_model(path="model_checkpoint.pth", device='cpu'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Aucun checkpoint à {path}")
    checkpoint = torch.load(path, map_location=device)
    model = PPOActorCritic()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint.get('episode', 0)
    print(f"[MODEL] Chargé épisode {episode} depuis {path}")
    return model, optimizer, episode
