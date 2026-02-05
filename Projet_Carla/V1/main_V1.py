import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import gc
import time

from simulation_V1 import CarlaEnv
from model_PPO_V1 import PPOActorCritic, RolloutBuffer, load_model, save_model, ppo_update


learning_rate = 3e-4

# Utility function to encode state
def encode_state(state_dict, device="cpu"):
    """
    Convertit le dict CARLA en tensor plat
    """
    state = np.concatenate([
        state_dict["lidar"],          # (32,)
        state_dict["collision"],      # (1,)
        state_dict["speed"],          # (1,) - Décommenté pour correspondre au modèle
        state_dict["lane_offset"],    # (1,)
        state_dict["lane_angle"],     # (1,)
        state_dict["goal_direction"], # (2,)
        state_dict["goal_distance"],  # (1,)
    ]).astype(np.float32)

    return torch.tensor(state, device=device).unsqueeze(0)

def safe_gpu_cleanup():
    """Nettoyage sécurisé de la mémoire GPU"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception as e:
        print(f"[GPU] Erreur nettoyage: {e}")

# ----------------------------
# Boucle principale d'entraînement
# ----------------------------

if __name__ == "__main__":
    # Variables d'environnement
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Utilisation de l'appareil : {device}")

    # Créer environnement
    env = CarlaEnv(host="localhost", port=2000, Num_sectors_lidar=16, lidar_range=50)

    # charger/créer modèle
    start_episode = 0
    model_charged = False
    model = None
    optimizer = None
    
    if os.path.exists("V1/model_checkpoint_PPO.pth"):
        try:
            model, optimizer, start_episode = load_model("V1/model_checkpoint_PPO.pth", device=device)
            model_charged = True
            print(f"[MODEL] Modèle chargé avec succès depuis l'épisode {start_episode}")
        except Exception as e:
            print(f"[MODEL] Erreur chargement, création nouveau modèle: {e}")
            model = None
    
    # Créer un nouveau modèle si le chargement a échoué ou si le fichier n'existe pas
    if model is None:
        model = PPOActorCritic().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print("[MODEL] Nouveau modèle créé")

    num_episodes = 1000
    df_rewards = []

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        total_reward_base = 0.0
        total_reward_lane_keeping = 0.0
        total_reward_consistency_bonus = 0.0
        total_reward_speed = 0.0
        total_reward_exploration = 0.0
        total_reward_collision = 0.0
        total_reward_immobility = 0.0
        total_reward_off_road = 0.0
        total_reward_timeout = 0.0
        total_reward_off_road_termination = 0.0

        buffer = RolloutBuffer()
        steps = 0
        collision_recovery = False
        
        while not done and steps < 1000:
            # Vérifier si collision critique détectée
            if env.critical_collision and not collision_recovery:
                print(f"[COLLISION] Collision critique détectée - arrêt épisode")
                collision_recovery = True
                # Nettoyage immédiat GPU
                safe_gpu_cleanup()
                # Forcer la fin de l'épisode
                done = True
                break
                
            state_tensor = encode_state(state, device)

            with torch.no_grad():
                action, log_prob, value = model.act(state_tensor)

            try:
                next_state, reward, done, reward_components = env.step(action)
            except Exception as e:
                print(f"[ERROR] Erreur durant step: {e}")
                # En cas d'erreur, forcer reset
                safe_gpu_cleanup()
                done = True
                break

            buffer.states.append(state_tensor.squeeze(0))
            buffer.actions.append(action.squeeze(0))
            buffer.log_probs.append(log_prob.item())
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.values.append(value.item())

            # Mettre à jour l'état et accumuler les récompenses
            state = next_state
            total_reward += reward
            total_reward_base += reward_components.get("base", 0.0)
            total_reward_lane_keeping += reward_components.get("lane_keeping", 0.0)
            total_reward_consistency_bonus += reward_components.get("consistency_bonus", 0.0)
            total_reward_speed += reward_components.get("speed", 0.0)
            total_reward_exploration += reward_components.get("exploration", 0.0)
            total_reward_collision += reward_components.get("collision", 0.0)
            total_reward_immobility += reward_components.get("immobility", 0.0)
            total_reward_off_road += reward_components.get("off_road", 0.0)
            # total_reward_timeout += reward_components.get("timeout", 0.0)
            total_reward_off_road_termination += reward_components.get("off_road_termination", 0.0)

            steps += 1

        # Mise à jour PPO seulement si on a des données valides
        if len(buffer.states) >= 256:
            try:
                ppo_update(model, optimizer, buffer)
            except Exception as e:
                print(f"[PPO] Erreur mise à jour: {e}")
                safe_gpu_cleanup()

        print(f"Iteration {episode} done")


        # Sauvegarder les récompenses de l'épisode
        df_rewards.append({
            "episode": episode,
            "reward": total_reward,
            "base": total_reward_base,
            "lane_keeping": total_reward_lane_keeping,
            "consistency_bonus": total_reward_consistency_bonus,
            "speed": total_reward_speed,
            "exploration": total_reward_exploration,
            "collision": total_reward_collision,
            "immobility": total_reward_immobility,
            "off_road": total_reward_off_road,
            # "timeout": total_reward_timeout,
            "off_road_termination": total_reward_off_road_termination,
        })

        # Sauvegarder le modèle tous les 10 épisodes
        if episode % 10 == 0:
            save_model(model, optimizer, episode)

        # Affichage des résultats de l'épisode
        print(f"[TRAIN] Episode {episode} | Reward: {total_reward:.2f}")
        print(

            f"[TRAIN] Components -> Base: {total_reward_base}, "
            f"Lane Keeping: {total_reward_lane_keeping:.2f}, "
            f"Consistency Bonus: {total_reward_consistency_bonus:.2f}, "
            f"Speed: {total_reward_speed:.2f}, "
            f"Exploration: {total_reward_exploration:.2f}, "
            f"Collision: {total_reward_collision:.2f}, "
            f"Immobility: {total_reward_immobility:.2f}, "
            f"Off Road: {total_reward_off_road:.2f}, "
            # f"Timeout: {total_reward_timeout:.2f}, "
            f"Off Road Termination: {total_reward_off_road_termination:.2f}, "
        )

        # Nettoyage de la mémoire GPU
        safe_gpu_cleanup()
        time.sleep(1)  # Pause pour permettre la libération de la mémoire

    # Conversion de df_rewards en DataFrame pandas pour le traçage
    df_rewards = pd.DataFrame(df_rewards)

    env.close()

    # Tracé des récompenses
    plt.figure(figsize=(12, 6))
    plt.plot(df_rewards['episode'], df_rewards['reward'], label='Total Reward')
    plt.plot(df_rewards['episode'], df_rewards['base'], label='Base')
    plt.plot(df_rewards['episode'], df_rewards['lane_keeping'], label='Lane Keeping')
    plt.plot(df_rewards['episode'], df_rewards['consistency_bonus'], label='Consistency Bonus')
    plt.plot(df_rewards['episode'], df_rewards['speed'], label='Speed')
    plt.plot(df_rewards['episode'], df_rewards['exploration'], label='Exploration')
    plt.plot(df_rewards['episode'], df_rewards['collision'], label='Collision')
    plt.plot(df_rewards['episode'], df_rewards['immobility'], label='Immobility')
    plt.plot(df_rewards['episode'], df_rewards['off_road'], label='Off Road')
    # plt.plot(df_rewards['episode'], df_rewards['timeout'], label='Timeout')
    plt.plot(df_rewards['episode'], df_rewards['off_road_termination'], label='Off Road Termination')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Récompenses par Épisode')
    plt.legend()  # Ajout de la légende
    plt.grid()
    plt.savefig("training_rewards_PPO.png")