# Projet PPO - Conduite Autonome avec CARLA

## 📋 Description

Ce projet implémente un système d'apprentissage par renforcement pour la conduite autonome dans le simulateur CARLA. Il utilise l'algorithme **Proximal Policy Optimization (PPO)** pour entraîner un véhicule à naviguer de manière autonome tout en respectant les règles de conduite.

## 🎯 Objectifs

- Apprendre à un véhicule à conduire de manière autonome
- Maintenir le véhicule au centre de sa voie
- Éviter les collisions avec les obstacles
- Maintenir une vitesse appropriée
- Progresser vers une destination définie

## 🏗️ Architecture du Projet

```
projet/
│
├── simulation_V6.py      # Environnement CARLA (gym)
├── model_PPO_V6.py       # Modèle PPO Actor-Critic
├── main_V6.py            # Script d'entraînement principal
├── requirements.txt      # Dépendances Python
└── V0/
    └── model_checkpoint_PPO.pth  # Checkpoints du modèle
```

## 🚀 Installation

### Prérequis

- **CARLA Simulator** (version 0.9.13 ou supérieure)
- **Python** 3.8+
- **CUDA** (optionnel, pour GPU)
- **8 Go RAM minimum** (16 Go recommandé)

### Étapes d'Installation

1. **Installer CARLA**
   ```bash
   # Télécharger depuis https://github.com/carla-simulator/carla/releases
   # Extraire et noter le chemin d'installation
   ```

2. **Configurer l'environnement Python**
   ```bash
   # Créer un environnement virtuel
   python -m venv carla_env
   source carla_env/bin/activate  # Linux/Mac
   # ou
   carla_env\Scripts\activate  # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer CARLA PythonAPI**
   ```bash
   # Ajouter le chemin vers CARLA PythonAPI
   export PYTHONPATH=$PYTHONPATH:/path/to/CARLA/PythonAPI/carla
   ```

## 🎮 Utilisation

### Démarrage du Serveur CARLA

```bash
# Dans le répertoire CARLA
./CarlaUE4.sh -RenderOffScreen -carla-port=2000 -nosound
# ou sur Windows
CarlaUE4.exe -RenderOffScreen -carla-port=2000 -nosound
```

### Entraînement du Modèle

```bash
python main_V6.py
```

Le script va :
- Charger un modèle existant si disponible (depuis `V0/model_checkpoint_PPO.pth`)
- Créer un nouveau modèle sinon
- Entraîner pendant 300 épisodes
- Sauvegarder les checkpoints tous les 3 épisodes
- Générer un graphique des récompenses (`training_rewards_PPO.png`)

### Paramètres Configurables

Dans `main_V6.py` :
- `num_episodes = 300` : Nombre d'épisodes d'entraînement
- `learning_rate = 3e-4` : Taux d'apprentissage
- `episode % 3 == 0` : Fréquence de sauvegarde

Dans `simulation_V6.py` :
- `Num_sectors_lidar=16` : Résolution du LIDAR
- `lidar_range=50` : Portée du LIDAR (mètres)
- `max_obstacles=3` : Nombre d'obstacles maximum

## 📊 Composants du Système de Récompense

Le système de récompense est multi-objectifs et équilibre plusieurs aspects :

| Composant | Poids | Description |
|-----------|-------|-------------|
| **Base** | +0.01 | Récompense de survie |
| **Lane Keeping** | +0.25 à +0.55 | Maintien dans la voie |
| **Consistency Bonus** | +0.05 à +0.2 | Conduite stable prolongée |
| **Speed** | ±0.5 | Vitesse appropriée |
| **Exploration** | +0.1 max | Distance parcourue |
| **Collision** | -500 | Pénalité collision |
| **Immobility** | -0.001/step | Pénalité immobilité |
| **Off-Road** | Variable | Conduite hors route |

## 🧠 Architecture du Réseau de Neurones

### Entrées (39 dimensions)
- **LIDAR** : 32 secteurs (distances normalisées)
- **Collision** : 1 intensité
- **Speed** : 1 vitesse actuelle
- **Lane Offset** : 1 décalage latéral
- **Lane Angle** : 1 angle avec la voie
- **Goal Direction** : 2 vecteur directionnel
- **Goal Distance** : 1 distance à l'objectif

### Architecture
```
Input (39) → FC(256) → ReLU → FC(256) → ReLU
                                        ↓
                        ┌───────────────┴───────────────┐
                        ↓                               ↓
                   Actor (3)                       Critic (1)
              (steer, throttle, brake)            (value estimate)
```

### Sorties
- **Actions** : 3 valeurs continues [0, 1]
  - Steering (direction)
  - Throttle (accélération)
  - Brake (freinage)

## 📈 Monitoring de l'Entraînement

Le script génère automatiquement :
- **Logs console** : Récompenses et composants par épisode
- **Graphique** : `training_rewards_PPO.png` avec toutes les composantes
- **Checkpoints** : Sauvegardés régulièrement dans `V0/`

## 🔧 Dépannage

### Problème : "Connection refused"
```bash
# Vérifier que CARLA est lancé
ps aux | grep Carla  # Linux/Mac
tasklist | findstr Carla  # Windows
```

### Problème : "CUDA Out of Memory"
```python
# Dans main_V6.py, réduire la taille du batch ou utiliser CPU
device = torch.device("cpu")
```

### Problème : Le véhicule reste immobile
- Vérifier que `min_throttle = 0` dans `simulation_V6.py`
- Le système force automatiquement le throttle si vitesse < 1 m/s

## 📝 Licence

Ce projet est fourni à des fins éducatives et de recherche.

## 👥 Contributeurs

Projet de recherche en apprentissage par renforcement appliqué à la conduite autonome.

## 📚 Références

- [CARLA Simulator](https://carla.org/)
- [PPO Algorithm (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [PyTorch Documentation](https://pytorch.org/docs/)
