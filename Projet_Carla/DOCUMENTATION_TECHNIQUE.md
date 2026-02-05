# Documentation Technique - Projet PPO CARLA V6

## Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Architecture Système](#architecture-système)
3. [Composants Détaillés](#composants-détaillés)
4. [Algorithme PPO](#algorithme-ppo)
5. [Environnement CARLA](#environnement-carla)
6. [Fonction de Récompense](#fonction-de-récompense)
7. [Optimisations et Considérations](#optimisations-et-considérations)

---

## 1. Vue d'Ensemble

### 1.1 Objectif du Projet

Développer un système de conduite autonome basé sur l'apprentissage par renforcement profond (Deep RL) utilisant l'algorithme Proximal Policy Optimization (PPO) dans l'environnement de simulation CARLA.

### 1.2 Stack Technique

- **Langage** : Python 3.8+
- **Framework RL** : PyTorch 2.0+
- **Simulateur** : CARLA 0.9.13+
- **API Gym** : Gymnasium
- **Accélération** : CUDA (optionnel)

### 1.3 Métriques de Performance

- **Récompense totale** : Somme des récompenses par épisode
- **Distance parcourue** : Distance maximale depuis le spawn
- **Lane keeping streak** : Nombre de steps consécutifs bien centrés
- **Collisions** : Intensité et fréquence
- **Vitesse moyenne** : Performance de déplacement

---

## 2. Architecture Système

### 2.1 Diagramme de Flux

```
┌─────────────────┐
│  CARLA Server   │
│   (Town01)      │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────┐
│    CarlaEnv (simulation_V6)     │
│  • Sensors (LIDAR, Collision)   │
│  • State Extraction             │
│  • Reward Calculation           │
│  • Episode Management           │
└────────┬────────────────────────┘
         │
         ↓
┌─────────────────────────────────┐
│   PPOActorCritic (model_PPO)    │
│  • Shared Backbone (256→256)    │
│  • Actor Head (mean + std)      │
│  • Critic Head (value)          │
└────────┬────────────────────────┘
         │
         ↓
┌─────────────────────────────────┐
│    Training Loop (main_V6)      │
│  • Rollout Collection           │
│  • GAE Computation              │
│  • PPO Update                   │
│  • Checkpoint Management        │
└─────────────────────────────────┘
```

### 2.2 Pipeline de Données

```
CARLA World State → Sensors → Feature Extraction → Neural Network → Actions → Vehicle Control → CARLA
                                                           ↓
                                                    Reward Signal
                                                           ↓
                                                    Buffer Storage
                                                           ↓
                                                     PPO Training
```

---

## 3. Composants Détaillés

### 3.1 simulation_V6.py - Environnement CARLA

#### 3.1.1 Classe CarlaEnv

**Héritage** : `gym.Env`

**Responsabilités** :
- Connexion au serveur CARLA
- Gestion du véhicule et des capteurs
- Calcul des observations
- Calcul des récompenses
- Gestion du cycle de vie des épisodes

#### 3.1.2 Capteurs Implémentés

**1. LIDAR (sensor.lidar.ray_cast)**
```python
Paramètres:
- Channels: 1 (plan horizontal)
- Range: 50 mètres
- Points par seconde: 56000
- Secteurs: 32 (16 * 2 bins)
- Rotation: -90° à +90° en yaw

Traitement:
- Binning par secteur angulaire
- Minimum des distances par secteur
- Normalisation [0, lidar_range]
```

**2. Collision (sensor.other.collision)**
```python
Données:
- Normal impulse (vecteur 3D)
- Intensité = ||normal_impulse||

Seuils:
- Collision critique: > 75.0
- Pénalité déclenchée: > 50.0
```

**3. Position & Orientation**
```python
Sources:
- Vehicle.get_transform()
- World.get_map().get_waypoint()

Calculs:
- Lane offset: Distance perpendiculaire au centre de voie
- Lane angle: Différence d'orientation véhicule/voie
- Goal direction: Vecteur normalisé vers destination
- Goal distance: Distance euclidienne 2D
```

#### 3.1.3 Espace d'Observation

```python
observation_space = {
    "lidar": Box(0, 50, shape=(32,)),           # Distances LIDAR
    "collision": Box(0, ∞, shape=(1,)),         # Intensité collision
    "speed": Box(0, 200, shape=(1,)),           # Vitesse (km/h)
    "position": Box(-∞, ∞, shape=(2,)),         # Position (x, y)
    "lane_offset": Box(-∞, ∞, shape=(1,)),      # Offset latéral normalisé
    "lane_angle": Box(-π, π, shape=(1,)),       # Angle avec voie normalisé
    "goal_direction": Box(-1, 1, shape=(2,)),   # Direction but (x, y)
    "goal_distance": Box(0, ∞, shape=(1,))      # Distance au but
}

Total: 39 dimensions
```

#### 3.1.4 Gestion des Obstacles

```python
Paramètres:
- max_obstacles = 3
- spawn_distance = [30, 60] mètres
- respawn_interval = 100 steps

Logique:
- Spawn aléatoire devant le véhicule
- Vitesse constante (autopilot)
- Despawn si hors de portée
- Respawn périodique
```

---

### 3.2 model_PPO_V6.py - Réseau de Neurones PPO

#### 3.2.1 Architecture PPOActorCritic

```python
class PPOActorCritic(nn.Module):
    
    Couches:
    ├── Backbone (partagé)
    │   ├── Linear(39 → 256)
    │   ├── ReLU
    │   ├── Linear(256 → 256)
    │   └── ReLU
    │
    ├── Actor
    │   ├── actor_mean: Linear(256 → 3)
    │   └── actor_log_std: Parameter(3)
    │
    └── Critic
        └── critic: Linear(256 → 1)
```

**Initialisation** :
```python
# Actor mean
nn.init.xavier_uniform_(actor_mean.weight, gain=0.1)
nn.init.zeros_(actor_mean.bias)

# Actor std
actor_log_std = ones(3) * 0.5  # std initiale ≈ 1.65
```

#### 3.2.2 Forward Pass

```python
def forward(state_tensor):
    """
    Input: (B, 39)
    
    1. Backbone: (B, 39) → (B, 256)
    2. Actor: (B, 256) → (B, 3) mean, (3,) std
    3. Critic: (B, 256) → (B, 1) → (B,) value
    
    Output: mean, std, value
    """
```

#### 3.2.3 Sampling d'Actions

```python
def act(state_tensor):
    """
    Distribution: Normal(mean, std)
    
    1. Sample: raw_action ~ N(μ, σ)
    2. Transform: action = sigmoid(raw_action) ∈ [0, 1]
    3. Log prob: log π(raw_action|state)
    
    Raison sigmoid:
    - Borne les actions dans [0, 1]
    - Nécessaire pour CARLA (throttle, brake, steer)
    """
```

**Note importante** : Le code utilise `sigmoid` pour la transformation, mais le commentaire et `evaluate()` mentionnent `tanh`. Il y a une incohérence à corriger :

```python
# Dans act():
action = torch.sigmoid(raw_action)  # Actuel

# Dans evaluate():
raw_action = torch.atanh(...)  # Inverse de tanh, pas sigmoid!

# CORRECTION NÉCESSAIRE dans evaluate():
raw_action = torch.logit(torch.clamp(action, 1e-7, 1-1e-7))
```

#### 3.2.4 RolloutBuffer

```python
class RolloutBuffer:
    """
    Stockage temporaire des expériences d'un épisode
    
    Données:
    - states: List[Tensor(39)]
    - actions: List[Tensor(3)]
    - log_probs: List[float]
    - rewards: List[float]
    - dones: List[bool]
    - values: List[float]
    """
```

---

### 3.3 Algorithme PPO

#### 3.3.1 Generalized Advantage Estimation (GAE)

```python
def compute_gae(rewards, values, dones, γ=0.99, λ=0.95):
    """
    Calcul récursif des avantages:
    
    δₜ = rₜ + γ·V(sₜ₊₁)·(1-dₜ) - V(sₜ)
    Aₜ = δₜ + γ·λ·(1-dₜ)·Aₜ₊₁
    
    Returns: Rₜ = Aₜ + V(sₜ)
    
    Paramètres:
    - γ (gamma): Discount factor (importance du futur)
    - λ (lambda): GAE parameter (bias-variance tradeoff)
    """
```

**Intuition** :
- **γ = 0.99** : Valorise fortement les récompenses futures
- **λ = 0.95** : Balance entre low bias (λ→1) et low variance (λ→0)

#### 3.3.2 PPO Update

```python
def ppo_update(model, optimizer, buffer, 
               clip_eps=0.2, value_coef=0.5, entropy_coef=0.05, epochs=10):
    """
    Objectif PPO:
    L = L_CLIP + c₁·L_VF - c₂·H
    
    Où:
    - L_CLIP: Clipped surrogate objective
    - L_VF: Value function loss
    - H: Entropy bonus
    
    Hyperparamètres:
    - clip_eps (ε): 0.2 → ratio ∈ [0.8, 1.2]
    - value_coef (c₁): 0.5
    - entropy_coef (c₂): 0.05
    - epochs: 10 (mini-batches sur les mêmes données)
    """
```

**Clipped Surrogate Objective** :
```python
ratio = exp(log π_new - log π_old)
surr1 = ratio · A
surr2 = clip(ratio, 1-ε, 1+ε) · A
L_CLIP = -min(surr1, surr2)
```

**Avantages de PPO** :
- Stabilité : Clipping empêche les mises à jour trop agressives
- Efficacité : Réutilise les données (10 epochs)
- Simplicité : Pas de contrainte KL explicite

---

## 4. Environnement CARLA

### 4.1 Configuration du Monde

```python
World: Town01
Mode: Synchronous (fixed_delta_seconds = 0.05s → 20 FPS)
Weather: Cloudiness=0, Precipitation=0, Sun=45°
```

### 4.2 Véhicule

```python
Blueprint: vehicle.tesla.model3
Spawn: Aléatoire parmi les spawn points
Destination: carla.Location(x=100, y=50, z=0)
```

### 4.3 Boucle de Simulation

```python
1. apply_control(throttle, brake, steer)
2. world.tick() ou wait_for_tick()
3. update_obstacles()
4. read_sensors() → observation
5. compute_reward() → reward, done
```

---

## 5. Fonction de Récompense

### 5.1 Système Multi-Objectifs

```python
total_reward = Σ [
    base_reward,
    lane_reward,
    consistency_bonus,
    speed_reward,
    exploration_reward,
    collision_penalty,
    immobility_penalty,
    off_road_penalty,
    off_road_termination_penalty
]

Clipping: [-150, 150]
```

### 5.2 Détail des Composants

#### 5.2.1 Base Reward
```python
base_reward = 0.01
# Récompense minimale pour chaque step survécu
```

#### 5.2.2 Lane Keeping Reward (Progressif)

```python
Seuils:
- excellent_offset ≤ 0.1
- good_offset ≤ 0.3
- bad_offset ≤ 0.7
- critical_offset ≤ 0.9

Seuils d'angle:
- excellent_angle ≤ 0.05
- good_angle ≤ 0.15
- bad_angle ≤ 0.4
- critical_angle ≤ 0.7

Récompenses:
- Excellent: 0.25 + min(streak * 0.02, 0.3) → max 0.55
- Good: 0.1 + min(streak * 0.01, 0.1) → max 0.2
- Acceptable: 0.025
- Mauvais: -2·offset² - 3·angle²
```

**Streak System** :
```python
if excellent or good:
    lane_keeping_streak += 1
    off_road_steps = 0
else:
    lane_keeping_streak = max(0, streak - 2)  # Décroissance rapide
    if off_road:
        off_road_steps += 1
```

#### 5.2.3 Consistency Bonus

```python
if streak ≥ 50: bonus = 0.2
elif streak ≥ 20: bonus = 0.1
elif streak ≥ 10: bonus = 0.05
else: bonus = 0.0
```

#### 5.2.4 Speed Reward (Conditionnel)

```python
if good_keeping or excellent_keeping:
    speed_reward = 0.5 · clip(speed/10, 0, 1)
else:
    speed_reward = -0.05 · clip(speed/10, 0, 1)  # Pénalité
```

**Logique** : Récompenser la vitesse uniquement si bien centré

#### 5.2.5 Exploration Reward

```python
distance_from_spawn = ||position - spawn_location||
exploration_reward = min(distance_from_spawn * 0.05, 0.1)
```

#### 5.2.6 Pénalités Critiques

```python
# Collision
if collision > 0:
    penalty = -500.0
    
# Immobilité
penalty = -0.001 · stationary_steps

# Off-Road
penalty = -0.1 · (off_road_steps^1.5)

# Termination Off-Road
if off_road_steps > 30 or offset > 0.95:
    penalty = -250.0
    done = True
```

### 5.3 Conditions de Terminaison

```python
done = (
    collision > 50.0 or
    elapsed ≥ 5000 steps or
    off_road_steps > 30 or
    lane_offset > 0.95
)
```

---

## 6. Optimisations et Considérations

### 6.1 Gestion Mémoire GPU

```python
def safe_gpu_cleanup():
    """
    Appelé après chaque épisode
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
```

**Raisons** :
- CARLA + PyTorch peuvent saturer la VRAM
- Prévient les `CUDA Out of Memory`
- Crucial pour entraînements longs

### 6.2 Gestion des Collisions Critiques

```python
if collision_intensity > 75.0:
    self.critical_collision = True
    
# Dans main_V6:
if env.critical_collision:
    safe_gpu_cleanup()
    # Option: forcer done = True
```

### 6.3 Anti-Immobilité

```python
# Dans step():
if current_speed < 1.0:  # m/s
    control.throttle = 0.5
    control.brake = 0.0
```

**Problème adressé** : Le modèle peut apprendre à rester immobile pour éviter les pénalités

### 6.4 Sauvegarde Incrémentale

```python
if episode % 3 == 0:
    save_model(model, optimizer, episode)
```

**Avantages** :
- Reprendre l'entraînement après crash
- Tester différentes checkpoints
- Analyse de la convergence

### 6.5 Normalisation des Avantages

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Effet** : Stabilise l'entraînement en normalisant la magnitude des gradients

---

## 7. Hyperparamètres Clés

### 7.1 Réseau de Neurones

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| hidden_dim | 256 | Capacité suffisante pour 39 inputs |
| learning_rate | 3e-4 | Standard pour PPO |
| xavier_gain | 0.1 | Initialisation conservatrice |
| log_std_init | 0.5 | std ≈ 1.65, exploration modérée |

### 7.2 PPO

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| clip_eps | 0.2 | Limite les mises à jour de policy |
| value_coef | 0.5 | Poids de la value loss |
| entropy_coef | 0.05 | Encourage l'exploration |
| epochs | 10 | Réutilisation des données |
| γ (gamma) | 0.99 | Discount factor |
| λ (lambda) | 0.95 | GAE parameter |

### 7.3 Environnement

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| lidar_sectors | 32 | Résolution angulaire |
| lidar_range | 50m | Portée du capteur |
| max_steps | 1000 | Limite par épisode |
| timeout | 5000 | Timeout en steps |
| min_ppo_buffer | 256 | Taille min pour update |

---

## 8. Problèmes Connus et Solutions

### 8.1 Incohérence Sigmoid/Tanh

**Problème** :
```python
# act() utilise sigmoid
action = torch.sigmoid(raw_action)

# evaluate() utilise atanh (inverse de tanh)
raw_action = torch.atanh(...)  # ERREUR!
```

**Solution** :
```python
# Choisir une transformation et l'inverse correcte:

# Option A: Tanh (recommandé pour actions [-1, 1])
action = torch.tanh(raw_action)
raw_action = torch.atanh(torch.clamp(action, -0.99, 0.99))

# Option B: Sigmoid (actuel, pour actions [0, 1])
action = torch.sigmoid(raw_action)
raw_action = torch.logit(torch.clamp(action, 1e-7, 1-1e-7))
```

### 8.2 Immobilité Persistante

**Symptôme** : Le véhicule reste bloqué

**Causes** :
1. Pénalité de vitesse trop forte
2. Récompense de lane keeping domine
3. Throttle non forcé

**Solutions implémentées** :
- Force throttle = 0.5 si vitesse < 1 m/s
- Pénalité d'immobilité progressive
- Speed reward conditionnel au lane keeping

### 8.3 Explosions de Gradient

**Symptôme** : Loss NaN, comportement erratique

**Solutions** :
- Normalisation des avantages
- Clipping PPO (ratio ∈ [0.8, 1.2])
- Reward clipping ([-150, 150])
- Learning rate modéré (3e-4)

---

## 9. Recommandations d'Amélioration

### 9.1 Court Terme

1. **Corriger l'incohérence sigmoid/atanh**
2. **Curriculum Learning** : Commencer sans obstacles
3. **Replay Buffer** : Stocker les meilleures trajectoires
4. **Wandb Integration** : Tracking avancé

### 9.2 Moyen Terme

1. **Multi-Task Learning** : Plusieurs destinations
2. **Attention Mechanism** : Pour le LIDAR
3. **Recurrent Policy** : LSTM pour mémoire temporelle
4. **Domain Randomization** : Varier météo, trafic, maps

### 9.3 Long Terme

1. **Vision-Based Policy** : Caméra + CNN
2. **Hierarchical RL** : High-level (navigation) + Low-level (control)
3. **Transfer Learning** : Sim-to-Real
4. **Multi-Agent** : Interaction avec autres véhicules

---

## 10. Annexes

### 10.1 Commandes CARLA Utiles

```bash
# Lancer avec paramètres custom
./CarlaUE4.sh -quality-level=Low -windowed -ResX=800 -ResY=600

# Mode spectateur
./CarlaUE4.sh -carla-port=2000 -opengl

# Logs
tail -f CarlaUE4/Saved/Logs/CarlaUE4.log
```

### 10.2 Debugging PyTorch

```python
# Gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Watchdog NaN
torch.autograd.set_detect_anomaly(True)

# Profiling
with torch.profiler.profile() as prof:
    ppo_update(...)
print(prof.key_averages().table())
```

### 10.3 Formules Clés

**Entropy** :
```
H = -Σ π(a|s) log π(a|s)
Pour Gaussian: H = 0.5 log(2πe σ²)
```

**KL Divergence (Gaussians)** :
```
KL(π_old || π_new) = log(σ_new/σ_old) + (σ_old² + (μ_old - μ_new)²)/(2σ_new²) - 0.5
```

**PPO Clipping** :
```
r(θ) = π_θ(a|s) / π_θ_old(a|s)
L^CLIP(θ) = 𝔼[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
```

---

## Conclusion

Ce projet démontre une implémentation complète d'un système de conduite autonome par RL. Les points forts incluent un système de récompense sophistiqué, une gestion robuste de l'environnement CARLA, et une architecture PPO standard mais efficace. Les axes d'amélioration prioritaires sont la correction de l'incohérence sigmoid/tanh et l'implémentation d'un curriculum learning pour accélérer la convergence.
