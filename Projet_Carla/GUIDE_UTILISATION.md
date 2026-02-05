# Guide d'Utilisation - Conduite Autonome PPO

## 📖 Table des Matières

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Utilisation Basique](#utilisation-basique)
4. [Utilisation Avancée](#utilisation-avancée)
5. [Visualisation des Résultats](#visualisation-des-résultats)
6. [Dépannage](#dépannage)

---

## 1. Installation

### 1.1 Prérequis Système

**Configuration Minimale** :
- OS : Windows 10/11, Ubuntu 18.04+, ou macOS
- RAM : 8 Go
- GPU : NVIDIA avec 4 Go VRAM (optionnel mais recommandé)
- Stockage : 50 Go disponibles

**Configuration Recommandée** :
- RAM : 16 Go
- GPU : NVIDIA RTX 2060 ou supérieur (8 Go VRAM)
- Stockage : 100 Go disponibles (SSD)

### 1.2 Installation de CARLA

#### Windows

1. Téléchargez CARLA depuis [GitHub Releases](https://github.com/carla-simulator/carla/releases)
   ```
   CARLA_0.9.13.zip (environ 8 Go)
   ```

2. Extrayez dans un répertoire :
   ```
   C:\CARLA_0.9.13\
   ```

3. Ajoutez le PythonAPI au PATH :
   ```powershell
   # PowerShell
   $env:PYTHONPATH += ";C:\CARLA_0.9.13\PythonAPI\carla"
   ```

#### Linux

```bash
# Télécharger
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz

# Extraire
tar -xzf CARLA_0.9.13.tar.gz -C ~/CARLA_0.9.13

# Ajouter au PATH
echo 'export PYTHONPATH=$PYTHONPATH:~/CARLA_0.9.13/PythonAPI/carla' >> ~/.bashrc
source ~/.bashrc
```

#### macOS

```bash
# Télécharger depuis GitHub Releases
# Extraire et configurer comme Linux
export PYTHONPATH=$PYTHONPATH:/Applications/CARLA_0.9.13/PythonAPI/carla
```

### 1.3 Configuration Python

```bash
# Créer un environnement virtuel
python -m venv carla_ppo_env

# Activer l'environnement
# Windows
carla_ppo_env\Scripts\activate
# Linux/Mac
source carla_ppo_env/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Installer CARLA Python API
cd /path/to/CARLA_0.9.13/PythonAPI/carla/dist
pip install carla-0.9.13-*.whl
```

### 1.4 Vérification de l'Installation

```python
# test_installation.py
import carla
import torch
import numpy as np
import gymnasium as gym

print(f"CARLA version: {carla.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 2. Configuration

### 2.1 Structure des Dossiers

```
projet_ppo/
│
├── simulation_V6.py
├── model_PPO_V6.py
├── main_V6.py
├── requirements.txt
│
├── V0/                          # Checkpoints du modèle
│   └── model_checkpoint_PPO.pth
│
├── logs/                        # Logs d'entraînement (à créer)
│   ├── rewards.csv
│   └── tensorboard/
│
└── outputs/                     # Graphiques et résultats (à créer)
    └── training_rewards_PPO.png
```

### 2.2 Paramètres Modifiables

#### Dans `main_V6.py` :

```python
# Ligne 10-11
learning_rate = 3e-4          # Taux d'apprentissage
                              # Plus bas = plus stable, plus lent
                              # Plus haut = plus rapide, moins stable

# Ligne 61
num_episodes = 300            # Nombre total d'épisodes
                              # Recommandé : 500-1000 pour convergence

# Ligne 105
if len(buffer.states) >= 256: # Taille minimale du buffer
                              # Plus grand = updates moins fréquentes mais plus stables

# Ligne 127
if episode % 3 == 0:          # Fréquence de sauvegarde
                              # Ajuster selon espace disque disponible
```

#### Dans `simulation_V6.py` :

```python
# Ligne 27-29
Num_sectors_lidar = 16        # Résolution LIDAR (16 = 32 bins)
                              # Plus = plus précis, plus coûteux
lidar_range = 50              # Portée LIDAR en mètres
                              # Ajuster selon scenario

# Ligne 56-59
max_obstacles = 3             # Nombre d'obstacles
obstacle_spawn_distance = [30, 60]  # Distance de spawn
obstacle_respawn_interval = 100     # Fréquence de respawn
```

#### Dans `model_PPO_V6.py` :

```python
# Ligne 15
hidden_dim = 256              # Taille des couches cachées
                              # Plus = plus de capacité, plus lent

# Ligne 85-88
gamma = 0.99                  # Discount factor (importance futur)
lam = 0.95                    # GAE lambda (bias-variance)

# Ligne 96-99
clip_eps = 0.2                # PPO clipping (stabilité)
value_coef = 0.5              # Poids value loss
entropy_coef = 0.05           # Exploration
epochs = 10                   # Époques par update
```

### 2.3 Configuration GPU vs CPU

```python
# Forcer CPU (si problèmes GPU)
device = torch.device("cpu")

# Utiliser GPU si disponible (défaut)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utiliser un GPU spécifique (multi-GPU)
device = torch.device("cuda:1")  # Utilise le 2ème GPU
```

---

## 3. Utilisation Basique

### 3.1 Lancer le Serveur CARLA

#### Windows

```powershell
# Ouvrir PowerShell dans le dossier CARLA
cd C:\CARLA_0.9.13

# Mode sans rendu (recommandé pour entraînement)
.\CarlaUE4.exe -RenderOffScreen -carla-port=2000 -nosound

# Mode avec visualisation
.\CarlaUE4.exe -carla-port=2000 -windowed -ResX=1280 -ResY=720
```

#### Linux

```bash
cd ~/CARLA_0.9.13

# Sans rendu
./CarlaUE4.sh -RenderOffScreen -carla-port=2000 -nosound &

# Avec visualisation
./CarlaUE4.sh -carla-port=2000 -windowed -ResX=1280 -ResY=720
```

**Vérification** : Vous devriez voir un message de démarrage du serveur dans le terminal.

### 3.2 Lancer l'Entraînement

```bash
# Activer l'environnement
source carla_ppo_env/bin/activate  # Linux/Mac
carla_ppo_env\Scripts\activate     # Windows

# Lancer l'entraînement
python main_V6.py
```

**Sortie attendue** :
```
[CARLA] Connecté à localhost:2000
[MAIN] Utilisation de l'appareil : cuda
[MODEL] Nouveau modèle créé
Iteration 0 done
[TRAIN] Episode 0 | Reward: -423.45
[TRAIN] Components -> Base: 7.8, Lane Keeping: -234.12, ...
[MODEL] Sauvegardé épisode 0 -> V0/model_checkpoint_PPO.pth
...
```

### 3.3 Reprendre un Entraînement

Le script reprend automatiquement depuis le dernier checkpoint si disponible :

```bash
# Vérifier qu'un checkpoint existe
ls V0/model_checkpoint_PPO.pth

# Lancer normalement
python main_V6.py
```

**Sortie** :
```
[MODEL] Modèle chargé avec succès depuis l'épisode 150
```

### 3.4 Arrêter l'Entraînement

```bash
# Ctrl+C dans le terminal
# Le modèle sera sauvegardé au prochain checkpoint (tous les 3 épisodes)

# Pour arrêter proprement CARLA (Linux)
pkill -9 Carla

# Windows : Task Manager → Terminer CarlaUE4.exe
```

---

## 4. Utilisation Avancée

### 4.1 Modification du Système de Récompense

Pour ajuster les poids des récompenses, éditez `simulation_V6.py` :

```python
# Ligne 600-656 : reward_components

# Exemple : Augmenter l'importance du lane keeping
lane_reward = 0.5 + min(self.lane_keeping_streak * 0.03, 0.5)  # Au lieu de 0.25

# Exemple : Réduire la pénalité de collision
collision_penalty = -250.0 if collision > 0.0 else 0.0  # Au lieu de -500.0

# Exemple : Encourager plus la vitesse
speed_reward = 1.0 * np.clip(current_speed / 10.0, 0.0, 1.0)  # Au lieu de 0.5
```

### 4.2 Changer de Map CARLA

```python
# Dans simulation_V6.py, ligne 83
self.world = self.client.load_world("Town01")

# Remplacer par :
self.world = self.client.load_world("Town03")
# Maps disponibles : Town01-Town10, Town10HD
```

### 4.3 Entraînement par Curriculum

```python
# Créer un script curriculum_training.py

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from simulation_V6 import CarlaEnv
from model_PPO_V6 import *
from main_V6 import *

# Phase 1 : Sans obstacles
env = CarlaEnv(host="localhost", port=2000)
env.max_obstacles = 0
train(env, episodes=100, save_path="V0/phase1_checkpoint.pth")

# Phase 2 : 1 obstacle
env.max_obstacles = 1
model, optimizer, _ = load_model("V0/phase1_checkpoint.pth")
train(env, episodes=100, start_episode=100, save_path="V0/phase2_checkpoint.pth")

# Phase 3 : 3 obstacles
env.max_obstacles = 3
model, optimizer, _ = load_model("V0/phase2_checkpoint.pth")
train(env, episodes=200, start_episode=200, save_path="V0/phase3_checkpoint.pth")
```

### 4.4 Évaluation du Modèle

```python
# eval_model.py

import torch
import numpy as np
from simulation_V6 import CarlaEnv
from model_PPO_V6 import load_model
from main_V6 import encode_state

# Charger le modèle
model, _, episode = load_model("V0/model_checkpoint_PPO.pth", device="cuda")
model.eval()  # Mode évaluation

# Créer environnement
env = CarlaEnv(host="localhost", port=2000)

# Tester sur N épisodes
num_eval_episodes = 10
results = []

for i in range(num_eval_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 1000:
        state_tensor = encode_state(state, device="cuda")
        
        with torch.no_grad():
            action, _, _ = model.act(state_tensor)
        
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
    
    results.append({
        "episode": i,
        "reward": total_reward,
        "steps": steps
    })
    print(f"Eval Episode {i}: Reward={total_reward:.2f}, Steps={steps}")

env.close()

# Statistiques
import pandas as pd
df = pd.DataFrame(results)
print("\n=== Résultats d'Évaluation ===")
print(f"Récompense moyenne : {df['reward'].mean():.2f} ± {df['reward'].std():.2f}")
print(f"Steps moyens : {df['steps'].mean():.0f} ± {df['steps'].std():.0f}")
```

### 4.5 Logging Avancé avec TensorBoard

```python
# Installer TensorBoard
pip install tensorboard

# Modifier main_V6.py pour ajouter logging

from torch.utils.tensorboard import SummaryWriter

# Après ligne 55
writer = SummaryWriter(log_dir="logs/tensorboard")

# Dans la boucle d'entraînement (après ligne 139)
writer.add_scalar("Reward/Total", total_reward, episode)
writer.add_scalar("Reward/Base", total_reward_base, episode)
writer.add_scalar("Reward/LaneKeeping", total_reward_lane_keeping, episode)
# ... etc pour chaque composant

# À la fin
writer.close()
```

```bash
# Visualiser dans TensorBoard
tensorboard --logdir=logs/tensorboard
# Ouvrir http://localhost:6006 dans le navigateur
```

### 4.6 Export pour Déploiement

```python
# export_model.py

import torch
from model_PPO_V6 import PPOActorCritic, load_model

# Charger le modèle entraîné
model, _, _ = load_model("V0/model_checkpoint_PPO.pth", device="cpu")
model.eval()

# Option 1 : Sauvegarder seulement les poids
torch.save(model.state_dict(), "deployed_model_weights.pth")

# Option 2 : Sauvegarder le modèle complet
torch.save(model, "deployed_model_full.pth")

# Option 3 : TorchScript (pour production)
dummy_input = torch.randn(1, 39)  # (batch_size, state_dim)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("deployed_model_traced.pt")

print("Modèle exporté avec succès!")
```

---

## 5. Visualisation des Résultats

### 5.1 Graphique Automatique

Le script génère automatiquement `training_rewards_PPO.png` à la fin de l'entraînement.

### 5.2 Analyse Personnalisée

```python
# analyze_training.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Si vous avez sauvegardé les données
df = pd.read_csv("logs/rewards.csv")  # À implémenter dans main_V6.py

# Calcul de moyennes mobiles
window = 10
df['reward_smooth'] = df['reward'].rolling(window=window).mean()

# Plot avancé
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1 : Récompense totale
axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3, label='Raw')
axes[0, 0].plot(df['episode'], df['reward_smooth'], label=f'MA-{window}')
axes[0, 0].set_title('Récompense Totale')
axes[0, 0].legend()
axes[0, 0].grid()

# Subplot 2 : Lane Keeping
axes[0, 1].plot(df['episode'], df['lane_keeping'])
axes[0, 1].set_title('Lane Keeping Reward')
axes[0, 1].grid()

# Subplot 3 : Collisions
axes[1, 0].plot(df['episode'], df['collision'])
axes[1, 0].set_title('Collision Penalty')
axes[1, 0].grid()

# Subplot 4 : Vitesse
axes[1, 1].plot(df['episode'], df['speed'])
axes[1, 1].set_title('Speed Reward')
axes[1, 1].grid()

plt.tight_layout()
plt.savefig("outputs/analysis_detailed.png", dpi=300)
plt.show()
```

### 5.3 Statistiques d'Entraînement

```python
# training_stats.py

import json

# À ajouter dans main_V6.py pour sauvegarder les stats
stats = {
    "total_episodes": num_episodes,
    "best_reward": max([ep['reward'] for ep in df_rewards]),
    "worst_reward": min([ep['reward'] for ep in df_rewards]),
    "mean_reward": np.mean([ep['reward'] for ep in df_rewards]),
    "std_reward": np.std([ep['reward'] for ep in df_rewards]),
    "final_model_episode": num_episodes - 1
}

with open("outputs/training_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

print(json.dumps(stats, indent=2))
```

---

## 6. Dépannage

### 6.1 Problèmes de Connexion CARLA

**Erreur** : `Connection refused`

**Solutions** :
```bash
# 1. Vérifier que CARLA est lancé
ps aux | grep Carla  # Linux
tasklist | findstr Carla  # Windows

# 2. Vérifier le port
netstat -an | grep 2000  # Linux
netstat -an | findstr 2000  # Windows

# 3. Relancer CARLA avec port explicite
./CarlaUE4.sh -carla-port=2000

# 4. Vérifier firewall (Windows)
# Autoriser CarlaUE4.exe dans le pare-feu
```

### 6.2 Problèmes GPU

**Erreur** : `CUDA out of memory`

**Solutions** :
```python
# 1. Réduire hidden_dim dans model_PPO_V6.py
hidden_dim = 128  # Au lieu de 256

# 2. Utiliser CPU
device = torch.device("cpu")

# 3. Nettoyer la mémoire plus souvent
# Dans main_V6.py, ligne 107, réduire le seuil
if len(buffer.states) >= 128:  # Au lieu de 256

# 4. Fermer les autres applications GPU
# (navigateurs, autres modèles, etc.)
```

**Erreur** : `CUDA driver version insufficient`

```bash
# Vérifier version CUDA
nvidia-smi

# Installer PyTorch compatible
# Pour CUDA 11.8
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 6.3 Véhicule Immobile

**Symptôme** : Le véhicule ne bouge pas

**Solutions** :
```python
# 1. Vérifier dans simulation_V6.py, ligne 679
min_throttle = 0.1  # Au lieu de 0

# 2. Augmenter le throttle forcé (ligne 693)
control.throttle = 0.7  # Au lieu de 0.5

# 3. Augmenter la pénalité d'immobilité
# Ligne 641
immobility_penalty = -0.01 * max(self.stationary_steps, 0)  # Au lieu de -0.001
```

### 6.4 Entraînement ne Converge Pas

**Symptômes** : Récompense stagne ou oscille

**Solutions** :
```python
# 1. Réduire learning rate
learning_rate = 1e-4  # Au lieu de 3e-4

# 2. Augmenter exploration
# Dans model_PPO_V6.py, ligne 30
self.actor_log_std = nn.Parameter(torch.ones(action_dim) * 1.0)  # Au lieu de 0.5

# 3. Réduire clip_eps pour plus de stabilité
clip_eps = 0.1  # Au lieu de 0.2

# 4. Augmenter epochs PPO
epochs = 20  # Au lieu de 10

# 5. Implémenter curriculum learning (voir section 4.3)
```

### 6.5 Erreurs Python

**Erreur** : `ModuleNotFoundError: No module named 'carla'`

```bash
# Installer CARLA Python API
cd /path/to/CARLA/PythonAPI/carla/dist
pip install carla-0.9.13-*.whl

# Vérifier installation
python -c "import carla; print(carla.__version__)"
```

**Erreur** : `AttributeError: 'torch.sigmoid'`

```python
# Utiliser torch.sigmoid correctement
action = torch.sigmoid(raw_action)
# Pas: action = torch.sigmoid(raw_action())
```

### 6.6 Performance Lente

**Symptômes** : < 1 FPS, entraînement très lent

**Solutions** :
```bash
# 1. Utiliser mode sans rendu
./CarlaUE4.sh -RenderOffScreen

# 2. Réduire qualité graphique (si avec rendu)
./CarlaUE4.sh -quality-level=Low

# 3. Réduire résolution LIDAR
Num_sectors_lidar = 8  # Au lieu de 16

# 4. Désactiver obstacles
env.max_obstacles = 0

# 5. Utiliser GPU si disponible
device = torch.device("cuda")
```

### 6.7 Crash Aléatoires

**Symptômes** : Le script plante sans raison claire

**Solutions** :
```python
# 1. Augmenter les timeouts
# Dans simulation_V6.py, ligne 82
self.client.set_timeout(30.0)  # Au lieu de 10.0

# 2. Ajouter gestion d'erreurs
# Entourer world.tick() de try-except

# 3. Nettoyer mémoire plus souvent
# Appeler safe_gpu_cleanup() tous les 5 épisodes

# 4. Redémarrer CARLA périodiquement
# Après chaque 50 épisodes, script séparé
```

---

## 7. FAQ

**Q : Combien de temps prend l'entraînement ?**
A : Environ 10-20 heures pour 300 épisodes sur GPU RTX 3060. Dépend du hardware et des paramètres.

**Q : Puis-je entraîner sur CPU ?**
A : Oui, mais 5-10x plus lent. Recommandé uniquement pour tests rapides.

**Q : Le modèle peut-il conduire sur de vraies routes ?**
A : Non, il est entraîné sur CARLA. Transfer sim-to-real nécessite fine-tuning et domain adaptation.

**Q : Comment comparer différents modèles ?**
A : Utilisez le script d'évaluation (section 4.4) avec les mêmes seeds aléatoires.

**Q : Puis-je utiliser plusieurs GPUs ?**
A : Oui, mais nécessite modification du code pour DataParallel ou DistributedDataParallel.

**Q : Où trouver de l'aide supplémentaire ?**
A : 
- [Documentation CARLA](https://carla.readthedocs.io/)
- [Forum CARLA](https://github.com/carla-simulator/carla/discussions)
- [PyTorch Forums](https://discuss.pytorch.org/)

---

## 8. Checklist Pré-Entraînement

Avant de lancer un entraînement long :

- [ ] CARLA serveur lancé et accessible
- [ ] Python environment activé avec toutes dépendances
- [ ] GPU détecté (si applicable) : `torch.cuda.is_available()`
- [ ] Dossier `V0/` créé pour checkpoints
- [ ] Paramètres vérifiés et ajustés si nécessaire
- [ ] Espace disque suffisant (>10 Go)
- [ ] Checkpoint précédent sauvegardé si modification majeure
- [ ] Test court (10 épisodes) sans erreurs

**Commande complète** :
```bash
# Terminal 1 : CARLA
./CarlaUE4.sh -RenderOffScreen -carla-port=2000 -nosound

# Terminal 2 : Entraînement
source carla_ppo_env/bin/activate
python main_V6.py

# Terminal 3 : Monitoring (optionnel)
watch -n 5 'tail -n 20 logs/training.log'
```

---

## Conclusion

Ce guide couvre les cas d'usage principaux du projet. Pour des besoins spécifiques ou des problèmes non couverts, consultez la documentation technique (DOCUMENTATION_TECHNIQUE.md) ou créez une issue sur le repository du projet.

Bonne chance avec votre entraînement ! 🚗🤖
