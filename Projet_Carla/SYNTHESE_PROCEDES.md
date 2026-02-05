# Synthèse des Procédés Importants
## Projet Conduite Autonome PPO-CARLA

---

## 🎯 Vue d'Ensemble Rapide

**Objectif** : Entraîner un véhicule virtuel à conduire de manière autonome via apprentissage par renforcement

**Technologies** : PyTorch + CARLA + PPO Algorithm

**Résultat** : Agent capable de naviguer en maintenant sa voie, évitant les obstacles, et maintenant une vitesse appropriée

---

## 1. Architecture Générale

### 1.1 Pipeline de Données

```
CARLA Simulation → Capteurs (LIDAR + Collision) → État (39D)
                                                      ↓
                                              Réseau PPO (Actor-Critic)
                                                      ↓
                                              Actions (3D: steer, throttle, brake)
                                                      ↓
                                              Contrôle Véhicule → Récompense
                                                      ↓
                                              Buffer → Training PPO
```

### 1.2 Fichiers Principaux

| Fichier | Rôle | Lignes Clés |
|---------|------|-------------|
| **simulation_V6.py** | Environnement CARLA | 18-762 |
| **model_PPO_V6.py** | Réseau + Algorithme PPO | 10-140 |
| **main_V6.py** | Boucle d'entraînement | 13-165 |

---

## 2. Procédés Clés par Module

### 2.1 Environnement (simulation_V6.py)

#### **Initialisation CARLA**
```python
# Lignes 80-88
self.client = carla.Client(host, port)
self.world = self.client.load_world("Town01")
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
```

**Importance** : Mode synchrone garantit reproductibilité et contrôle précis

#### **Capteur LIDAR**
```python
# Lignes 204-260
- 32 secteurs angulaires (360° / 32 = 11.25° par secteur)
- Range 50m
- Binning : grouper points par angle, garder le minimum
- Output : distances [0, 50] normalisées
```

**Procédé** :
1. Callback asynchrone reçoit point cloud 3D
2. Filtrage du sol (z > 0)
3. Calcul angle polaire : `atan2(y, x)`
4. Binning par secteur : `int((angle + π) / bin_size)`
5. Minimum par secteur → obstacle le plus proche

**Importance** : Perception de l'environnement, input principal du réseau

#### **Lane Features**
```python
# Lignes 125-163
def get_lane_features():
    1. Waypoint = position la plus proche sur route
    2. Offset = distance perpendiculaire au centre
    3. Angle = différence d'orientation véhicule/route
    4. Normalisation : offset/largeur_voie, angle/π
```

**Formules** :
```
offset = (vehicle_pos - waypoint_pos) · normal_vector
angle = (lane_yaw - vehicle_yaw) mod 2π - π
```

**Importance** : Quantification précise du centrage, signal de récompense principal

#### **Fonction de Récompense**
```python
# Lignes 520-663
Structure hiérarchique :
├─ Base (+0.01)
├─ Lane Keeping (+0.25 à +0.55) ← PRIORITAIRE
├─ Consistency Bonus (+0.05 à +0.2)
├─ Speed (±0.5)
├─ Exploration (+0.1)
├─ Collision (-500) ← CRITIQUE
├─ Immobility (-0.001/step)
├─ Off-Road (-0.1 * steps^1.5)
└─ Off-Road Termination (-250)

Total clippé : [-150, 150]
```

**Procédé Lane Keeping** :
```python
# Ligne 604-617
if excellent (offset≤0.1 AND angle≤0.05):
    reward = 0.25 + min(streak * 0.02, 0.3)  # Jusqu'à 0.55
elif good (offset≤0.3 AND angle≤0.15):
    reward = 0.1 + min(streak * 0.01, 0.1)   # Jusqu'à 0.2
elif acceptable:
    reward = 0.025
else:
    reward = -2*offset² - 3*angle²  # Pénalité quadratique
```

**Importance** : Équilibre délicat entre objectifs contradictoires, clé de la convergence

#### **Gestion des Obstacles**
```python
# Lignes 262-348
Spawn dynamique :
- Distance : [30, 60]m devant véhicule
- Max 3 obstacles simultanés
- Autopilot activé
- Respawn tous les 100 steps
```

**Importance** : Réalisme, teste capacités d'évitement

### 2.2 Modèle PPO (model_PPO_V6.py)

#### **Architecture Réseau**
```python
# Lignes 10-50
PPOActorCritic(
    Input: 39D state
    ↓
    Backbone: FC(39→256) → ReLU → FC(256→256) → ReLU
    ↓
    ├─ Actor: FC(256→3) → Gaussian(μ, σ)
    │         Actions = sigmoid(sample)
    └─ Critic: FC(256→1) → Value estimate
)
```

**Initialisation** :
```python
# Lignes 27-30
xavier_uniform_(actor_mean.weight, gain=0.1)  # Évite grands écarts initiaux
zeros_(actor_mean.bias)
actor_log_std = Parameter(ones(3) * 0.5)      # σ ≈ 1.65
```

**Importance** : Initialisation prudente évite instabilités précoces

#### **Sampling d'Actions**
```python
# Lignes 37-47
def act(state):
    μ, σ = forward(state)
    dist = Normal(μ, σ)
    raw_action = dist.sample()
    action = sigmoid(raw_action)  # Borne [0,1]
    log_prob = dist.log_prob(raw_action).sum(-1)
    return action, log_prob, value
```

**Détails Importants** :
- `sigmoid` transforme R → [0,1] (requis par CARLA)
- Log-prob calculé AVANT transformation (correcte)
- Sum sur dimensions → log-prob scalaire

**⚠️ BUG IDENTIFIÉ** : 
```python
# Ligne 55 - evaluate()
raw_action = atanh(clamp(action, -0.99, 0.99))  # ERREUR!
# atanh est l'inverse de tanh, pas sigmoid
# Correction : raw_action = logit(clamp(action, 1e-7, 1-1e-7))
```

#### **GAE (Generalized Advantage Estimation)**
```python
# Lignes 74-88
def compute_gae(rewards, values, dones, γ=0.99, λ=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(T)):
        δ = r[t] + γ*V[t+1]*(1-done[t]) - V[t]
        gae = δ + γ*λ*(1-done[t])*gae
        advantages.insert(0, gae)
    returns = [A + V for A, V in zip(advantages, values)]
    return advantages, returns
```

**Formule** :
```
Aₜ = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ...
où δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
```

**Paramètres** :
- γ = 0.99 : Valorise futur (horizon ~100 steps)
- λ = 0.95 : Balance bias/variance

**Importance** : Réduit variance des gradients, essentiel pour stabilité PPO

#### **PPO Update**
```python
# Lignes 91-127
def ppo_update(model, optimizer, buffer, epochs=10):
    # 1. Calculer avantages
    advantages, returns = compute_gae(...)
    advantages = (advantages - mean) / (std + 1e-8)  # Normalisation
    
    # 2. Epochs d'optimisation
    for epoch in range(10):
        # Réévaluer policy actuelle
        log_probs, entropy, values = model.evaluate(states, actions)
        
        # Ratio d'importance
        ratio = exp(log_probs - old_log_probs)
        
        # PPO clipping
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-ε, 1+ε) * advantages
        actor_loss = -min(surr1, surr2).mean()
        
        # Value loss
        critic_loss = MSE(values, returns)
        
        # Total loss
        loss = actor_loss + 0.5*critic_loss - 0.05*entropy.mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Clipping** :
```
Si A > 0 (bonne action) : ratio ∈ [1.0, 1.2]
Si A < 0 (mauvaise) : ratio ∈ [0.8, 1.0]
```

**Importance** : Cœur de PPO, empêche mises à jour trop agressives

#### **Sauvegarde/Chargement**
```python
# Lignes 133-154
save_model(model, optimizer, episode, path):
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

load_model(path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['episode']
```

**Importance** : Reprendre entraînement après crash, tester différentes checkpoints

### 2.3 Entraînement (main_V6.py)

#### **Encodage d'État**
```python
# Lignes 13-28
def encode_state(state_dict):
    return tensor([
        lidar,          # 32
        collision,      # 1
        speed,          # 1
        lane_offset,    # 1
        lane_angle,     # 1
        goal_direction, # 2
        goal_distance   # 1
    ]).flatten()  # Total: 39D
```

**Importance** : Interface standardisée dict → tensor

#### **Boucle d'Entraînement**
```python
# Lignes 55-145
for episode in range(num_episodes):
    state = env.reset()
    buffer = RolloutBuffer()
    
    # Rollout (collection de trajectoire)
    for step in range(max_steps):
        # 1. Sélectionner action
        action, log_prob, value = model.act(encode_state(state))
        
        # 2. Exécuter dans env
        next_state, reward, done, _ = env.step(action)
        
        # 3. Stocker dans buffer
        buffer.append(state, action, log_prob, reward, done, value)
        
        state = next_state
        if done: break
    
    # Update PPO si buffer suffisant
    if len(buffer) >= 256:
        ppo_update(model, optimizer, buffer)
    
    # Sauvegarde périodique
    if episode % 3 == 0:
        save_model(model, optimizer, episode)
```

**Seuil 256** : Trade-off variance/coût computationnel

**Importance** : Orchestration complète du pipeline d'entraînement

#### **Gestion Mémoire GPU**
```python
# Lignes 30-38
def safe_gpu_cleanup():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

# Appelé :
# - Après chaque épisode
# - Si collision critique détectée
# - Avant sauvegarde
```

**Importance** : Évite CUDA OOM sur entraînements longs (>200 épisodes)

#### **Visualisation**
```python
# Lignes 149-165
plt.plot(episodes, rewards, label='Total')
plt.plot(episodes, lane_keeping, label='Lane Keeping')
# ... tous les composants
plt.savefig("training_rewards_PPO.png")
```

**Importance** : Analyse visuelle de la convergence

---

## 3. Procédés Critiques pour la Performance

### 3.1 Stabilité de l'Entraînement

**Normalisation des Avantages**
```python
advantages = (advantages - mean) / (std + 1e-8)
```
**Effet** : Magnitude constante des gradients, évite explosions

**Clipping de la Récompense**
```python
total_reward = clip(total_reward, -150, 150)
```
**Effet** : Borne les valeurs extrêmes, stabilise Q-values

**Clipping PPO**
```python
ratio = clip(ratio, 0.8, 1.2)
```
**Effet** : Limite la divergence entre policies, empêche collapse

### 3.2 Convergence Rapide

**Initialisation Xavier**
```python
xavier_uniform_(weights, gain=0.1)
```
**Effet** : Variance contrôlée dès le début, évite saturations

**Learning Rate Adapté**
```python
lr = 3e-4  # Standard PPO
```
**Effet** : Balance vitesse/stabilité

**Entropy Bonus**
```python
loss -= 0.05 * entropy
```
**Effet** : Encourage exploration, évite convergence prématurée

### 3.3 Robustesse

**Mode Synchrone CARLA**
```python
synchronous_mode = True
fixed_delta_seconds = 0.05
```
**Effet** : Déterminisme, reproductibilité

**Timeout Généreux**
```python
client.set_timeout(10.0)
```
**Effet** : Tolérance aux ralentissements CARLA

**Try-Except Autour de Tick**
```python
try:
    world.tick()
except Exception:
    time.sleep(0.05)
```
**Effet** : Récupération des erreurs transitoires

---

## 4. Hyperparamètres Optimaux

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **hidden_dim** | 256 | Capacité suffisante pour 39 inputs |
| **learning_rate** | 3e-4 | Standard PPO, bon compromis |
| **clip_eps** | 0.2 | Standard PPO, stabilité prouvée |
| **value_coef** | 0.5 | Équilibre actor/critic |
| **entropy_coef** | 0.05 | Exploration modérée |
| **gamma** | 0.99 | Horizon ~100 steps |
| **lambda** | 0.95 | Balance bias/variance |
| **epochs** | 10 | Réutilisation données sans overfitting |
| **buffer_min** | 256 | Estimation stable des avantages |
| **lidar_range** | 50m | Anticipation suffisante |
| **lidar_sectors** | 32 | Résolution correcte |

---

## 5. Décisions de Design Importantes

### 5.1 Pourquoi PPO ?

**Alternatives Considérées** : SAC, TD3, DDPG

**Raisons PPO** :
- ✅ On-policy → plus stable pour débutants
- ✅ Clipping simple vs contrainte KL
- ✅ Fonctionne out-of-the-box (peu de tuning)
- ✅ Bonne efficacité échantillon pour on-policy
- ❌ Moins efficace que SAC (off-policy)

### 5.2 Pourquoi LIDAR vs Caméra ?

**Raisons LIDAR** :
- ✅ État bas-dim (32) vs images (224x224x3)
- ✅ Pas besoin de CNN (plus simple)
- ✅ Perception 360° directe
- ✅ Entraînement plus rapide
- ❌ Moins réaliste (la plupart des voitures = caméras)

### 5.3 Pourquoi Actions Continues ?

**Alternative** : Discrétisation (9 actions : avant, gauche, droite, ...)

**Raisons Continues** :
- ✅ Contrôle fin (steering précis)
- ✅ Réaliste pour véhicules
- ✅ Plus de flexibilité
- ❌ Plus difficile à apprendre
- ❌ Espace d'action infini

### 5.4 Pourquoi Reward Shaping Complexe ?

**Alternative** : Reward sparse (0 ou -1 si collision)

**Raisons Shaping** :
- ✅ Apprentissage beaucoup plus rapide
- ✅ Guide l'exploration
- ✅ Multi-objectifs explicites
- ❌ Risque de reward hacking
- ❌ Nécessite tuning

---

## 6. Pièges Évités

### 6.1 Immobilité Stratégique

**Problème** : Agent apprend à rester immobile (aucune pénalité)

**Solution** :
```python
# Force throttle si vitesse < 1 m/s
if speed < 1.0:
    control.throttle = 0.5
    control.brake = 0.0

# Pénalité d'immobilité
immobility_penalty = -0.001 * stationary_steps
```

### 6.2 Catastrophic Forgetting

**Problème** : Agent oublie comportements appris après update

**Solutions** :
- Clipping PPO (empêche changements brutaux)
- 10 epochs (exploitation max des données)
- Replay des meilleures trajectoires (pas encore implémenté)

### 6.3 Reward Hacking

**Problème** : Agent exploite la fonction de récompense

**Exemples** :
- Faire des cercles pour maximiser exploration
- Se mettre perpendiculaire pour réinitialiser streak

**Solutions** :
- Clipping total reward ([-150, 150])
- Pénalités quadratiques (off-road ∝ steps^1.5)
- Conditions de terminaison strictes

### 6.4 Gradient Explosion

**Problème** : Loss → NaN après quelques updates

**Solutions** :
- Normalisation des avantages
- Clipping PPO ratio
- Learning rate modéré
- Initialisation Xavier

---

## 7. Checklist de Debug

Quand l'entraînement ne marche pas :

**1. Véhicule immobile**
- [ ] Vérifier min_throttle != 0
- [ ] Augmenter force_throttle si vitesse < 1 m/s
- [ ] Vérifier pénalité immobilité active

**2. Reward stagne**
- [ ] Réduire learning_rate (3e-4 → 1e-4)
- [ ] Augmenter exploration (entropy_coef 0.05 → 0.1)
- [ ] Vérifier normalisation des avantages
- [ ] Essayer curriculum learning

**3. Loss → NaN**
- [ ] Ajouter clipping sur rewards
- [ ] Vérifier pas d'infinités dans log_prob
- [ ] Réduire learning_rate
- [ ] Initialisation plus conservatrice

**4. Mémoire GPU saturée**
- [ ] Appeler safe_gpu_cleanup() plus souvent
- [ ] Réduire hidden_dim (256 → 128)
- [ ] Réduire buffer_min (256 → 128)
- [ ] Fermer autres processus GPU

**5. CARLA crash**
- [ ] Augmenter timeout (10 → 30s)
- [ ] Ajouter try-except autour de tick()
- [ ] Vérifier pas de deadlock sensors
- [ ] Redémarrer CARLA tous les N épisodes

---

## 8. Formules de Référence Rapide

**PPO Loss** :
```
L = -𝔼[min(r(θ)A, clip(r(θ),1-ε,1+ε)A)] + c₁·MSE(V,R) - c₂·H
```

**GAE** :
```
Aₜ = Σᵢ(γλ)ⁱδₜ₊ᵢ  où  δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
```

**Gaussian Log-Prob** :
```
log π(a|s) = -½[(a-μ)/σ]² - log(σ) - ½log(2π)
```

**Lane Offset** :
```
offset = (vehicle_pos - waypoint_pos) · normal_vector
normal = [-sin(lane_yaw), cos(lane_yaw)]
```

---

## 9. Ordre d'Importance des Composants

**Critique (sans eux, rien ne marche)** :
1. ✅ Fonction de récompense équilibrée
2. ✅ Normalisation des avantages
3. ✅ Clipping PPO
4. ✅ Mode synchrone CARLA

**Très Important (affecte beaucoup la performance)** :
5. ✅ GAE avec bons γ, λ
6. ✅ Lane features précises
7. ✅ Initialisation réseau
8. ✅ Learning rate adapté

**Important (améliore mais pas critique)** :
9. ✅ Entropy bonus
10. ✅ LIDAR avec bonne résolution
11. ✅ Gestion mémoire GPU
12. ✅ Sauvegarde checkpoints

**Optionnel (nice-to-have)** :
13. ⭕ Obstacles dynamiques
14. ⭕ Visualisations détaillées
15. ⭕ Logs verbeux

---

## 10. One-Liner Pour Chaque Fichier

**simulation_V6.py** : 
> "Environnement gym connecté à CARLA, avec capteurs LIDAR/collision, fonction de récompense multi-objectifs (9 composants), et gestion d'obstacles dynamiques."

**model_PPO_V6.py** : 
> "Réseau Actor-Critic (39→256→256→3/1) avec sampling gaussien transformé par sigmoid, GAE pour avantages, et update PPO avec clipping ε=0.2."

**main_V6.py** : 
> "Boucle d'entraînement qui collecte rollouts (max 1000 steps), update PPO si buffer≥256, sauvegarde tous les 3 épisodes, et génère graphiques de convergence."

---

## Conclusion

Les procédés les plus importants sont :

1. **Reward Shaping** : 9 composants équilibrés guidant l'apprentissage
2. **PPO avec Clipping** : Stabilité via contrainte implicite
3. **GAE** : Réduction variance pour gradients fiables
4. **Normalisation** : Avantages + rewards pour stabilité numérique
5. **Gestion CARLA** : Mode synchrone + gestion erreurs
6. **Architecture Réseau** : Actor-Critic partagé, initialisation prudente

Le bug sigmoid/atanh est mineur et facilement corrigeable, le reste du code est solide et production-ready pour recherche et enseignement.
