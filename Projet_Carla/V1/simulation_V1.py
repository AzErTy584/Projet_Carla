import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from gymnasium import spaces
import gymnasium as gym
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import gc  # Pour le garbage collection

import carla  # nécessite CARLA PythonAPI accessible dans PYTHONPATH

# CarlaUE4.exe -RenderOffScreen -carla-port=2000 -nosound

class CarlaEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, host="localhost", port=2000, Num_sectors_lidar=16, lidar_range=50):
        super().__init__()

        # Paramètres
        self.host = host
        self.port = port
        self.Num_sector_lidar = Num_sectors_lidar
        self.lidar_bins = Num_sectors_lidar*2
        self.lidar_range = float(lidar_range)

        # Variables d'acteur
        self.client = None
        self.world = None 
        self.vehicle = None
        self.collision_sensor = None
        self.cam_sensor = None
        self.lidar_sensor = None
        self.position_sensor = None
        self.spectator = None
        
        # Flag pour collision critique
        self.critical_collision = False
        
        # Variables pour tracking de progression
        self.previous_distance_to_goal = None
        self.previous_position = None
        self.stationary_steps = 0
        self.max_distance_from_spawn = 0.0
        
        # Variables pour lane keeping tracking
        self.lane_keeping_streak = 0  # Compteur de bonne conduite consécutive
        self.off_road_steps = 0       # Compteur de conduite hors route

        # Données capteurs par défaut
        self.collision_intensity = 0.0
        self.lidar_data = np.zeros((self.lidar_bins,), dtype=np.float32)
        self.position = np.zeros((3,), dtype=np.float32)

        # Destination (à adapter)
        self.spawn_location = None
        self.final_destination = carla.Location(x=100.0, y=50.0, z=0.0)


        # Connexion au serveur CARLA
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.load_world("Town03")
            self.spectator = self.world.get_spectator()
            print(f"[CARLA] Connecté à {self.host}:{self.port}")
        except Exception as e:
            print(f"[CARLA] Erreur de connexion : {e}")
            raise

        # Réglages météo (exemple)
        try:
            weather = carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, sun_altitude_angle=45.0)
            self.world.set_weather(weather)
        except Exception:
            pass

        # Observation space
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(low=0.0, high=self.lidar_range, shape=(self.lidar_bins,), dtype=np.float32),
            "collision": spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
            "speed": spaces.Box(low=0.0, high=200.0, shape=(1,), dtype=np.float32),
            "position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "lane_offset": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "lane_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "goal_direction": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "goal_distance": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })

        # Si monde en mode synchrone, on devra appeler world.tick() nous-même
        self.sync = False
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            settings.fixed_delta_seconds = 1.0 / 20
            self.sync = settings.synchronous_mode
            print(f"[CARLA] Synchronous mode: {self.sync}")
        except Exception:
            self.sync = False


    # ----------------------------
    # Lane features helper and position
    # ----------------------------
    def get_lane_features(self):

        vehicle = self.vehicle
        world = self.world 
        # === Transform véhicule ===
        vt = vehicle.get_transform()
        loc = vt.location

        # === Waypoint ===
        wp = world.get_map().get_waypoint(
            loc,
            project_to_road=False
        )

        if wp is None:
            return 0.0, 0.0, np.array([0.0, 0.0], dtype=np.float32)

        # === Positions ===
        dx = loc.x - wp.transform.location.x
        dy = loc.y - wp.transform.location.y

        # === Angles ===
        car_yaw = np.deg2rad(vt.rotation.yaw)
        lane_yaw = np.deg2rad(wp.transform.rotation.yaw)

        # === Lane angle (normalisé) ===
        lane_angle = lane_yaw - car_yaw
        lane_angle = (lane_angle + np.pi) % (2 * np.pi) - np.pi
        lane_angle_norm = np.clip(lane_angle / np.pi, -1.0, 1.0)

        # === Lane offset (normalisé) ===
        normal_x = -np.sin(lane_yaw)
        normal_y =  np.cos(lane_yaw)
        lane_offset = dx * normal_x + dy * normal_y

        half_width = wp.lane_width * 0.5
        lane_offset_norm = np.clip(lane_offset / half_width, -1.0, 1.0)

        return lane_offset_norm, lane_angle_norm, [loc.x, loc.y]
    
    # ----------------------------
    # Camera / spectator helper
    # ----------------------------
    def _vehicule_camera(self):
        try:
            if self.vehicle and self.spectator:
                tf = self.vehicle.get_transform()
                # placer la caméra légèrement en arrière et au-dessus
                offset = carla.Location(x=-5.0, z=3.0)
                camera_pos = tf.transform(offset)
                self.spectator.set_transform(carla.Transform(camera_pos, tf.rotation))
        except Exception:
            pass

    # ----------------------------
    # Capteurs : attachement et callbacks
    # ----------------------------
    def _attach_collision_sensor(self):
        if self.collision_sensor is not None:
            return
        try:
            bp = self.world.get_blueprint_library().find("sensor.other.collision")
            transform = carla.Transform()
            self.collision_sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)

            def callback(event):
                try:
                    # normal_impulse est un vector
                    self.collision_intensity = float(event.normal_impulse.length())
                    if self.collision_intensity > 50.0:
                        self.critical_collision = True
                except Exception:
                    self.collision_intensity = 0.0

            self.collision_sensor.listen(callback)
        except Exception as e:
            print("[CARLA] Échec attach collision:", e)
            self.collision_sensor = None

    def _attach_lidar(self):
        if self.lidar_sensor is not None:
            return

        try:
            bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")

            bp.set_attribute("range", str(self.lidar_range))
            bp.set_attribute("rotation_frequency", "10")
            bp.set_attribute("points_per_second", "20000")
            bp.set_attribute("channels", "32")
            bp.set_attribute("upper_fov", "10")
            bp.set_attribute("lower_fov", "-30")

            transform = carla.Transform(carla.Location(z=2.5))
            self.lidar_sensor = self.world.spawn_actor(
                bp, transform, attach_to=self.vehicle
            )

            # ---- paramètres lidar sectorisé ----
            NUM_SECTORS = self.Num_sector_lidar
            MAX_RANGE = self.lidar_range

            def callback(point_cloud):
                try:
                    raw = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
                    if raw.size == 0:
                        self.lidar_data = np.ones(NUM_SECTORS * 2, dtype=np.float32)
                        return

                    points = raw.reshape((-1, 4))[:, :3]  # x, y, z

                    # Distances et angles
                    xy = points[:, :2]
                    dists = np.linalg.norm(xy, axis=1)
                    angles = np.arctan2(xy[:, 1], xy[:, 0])  # [-pi, pi]

                    # Normalisation angles -> [0, 2pi]
                    angles = (angles + 2 * np.pi) % (2 * np.pi)

                    # Initialisation secteurs
                    mins = np.full(NUM_SECTORS, MAX_RANGE, dtype=np.float32)
                    sums = np.zeros(NUM_SECTORS, dtype=np.float32)
                    counts = np.zeros(NUM_SECTORS, dtype=np.int32)

                    sector_width = 2 * np.pi / NUM_SECTORS

                    # Remplissage des secteurs
                    for dist, ang in zip(dists, angles):
                        if dist <= 0.0 or dist > MAX_RANGE:
                            continue
                        idx = int(ang // sector_width)
                        mins[idx] = min(mins[idx], dist)
                        sums[idx] += dist
                        counts[idx] += 1

                    # Moyennes
                    means = np.where(counts > 0, sums / counts, MAX_RANGE)

                    # Normalisation [0, 1]
                    mins /= MAX_RANGE
                    means /= MAX_RANGE

                    # Concaténation finale (32,)
                    self.lidar_data = np.concatenate([mins, means]).astype(np.float32)

                except Exception:
                    # Fallback sécurisé
                    self.lidar_data = np.ones(NUM_SECTORS * 2, dtype=np.float32)

            self.lidar_sensor.listen(callback)

        except Exception as e:
            print("[CARLA] Échec attach lidar:", e)
            self.lidar_sensor = None


    def _set_random_destination(self, min_dist=25.0, max_dist=35.0):
        transform = self.vehicle.get_transform()
        start = transform.location

        # Angle aléatoire
        angle = random.uniform(0, 2 * np.pi)

        # Distance aléatoire ou fixe
        dist = random.uniform(min_dist, max_dist)

        dx = dist * np.cos(angle)
        dy = dist * np.sin(angle)

        self.final_destination = carla.Location(
            x=start.x + dx,
            y=start.y + dy,
            z=start.z
        )

    def _get_goal_features(self):
        vehicle_loc = self.vehicle.get_transform().location
        dx = self.final_destination.x - vehicle_loc.x
        dy = self.final_destination.y - vehicle_loc.y
        distance = np.sqrt(dx**2 + dy**2)

        # Normalisation
        dx /= (distance + 1e-6)
        dy /= (distance + 1e-6)

        return np.array([dx, dy, distance], dtype=np.float32)

    # ----------------------------
    # Observation helper
    # ----------------------------    
    def _get_obs(self):
        # sécurité sur vehicle
        if self.vehicle is None:
            speed = np.array([0.0], dtype=np.float32)
            goal_features = np.array([0.0, 0.0, 100.0], dtype=np.float32)
            lane_offset, lane_angle, pos_xy = 0.0, 0.0, [0.0, 0.0]
        else:
            vel = self.vehicle.get_velocity()
            speed = [np.linalg.norm([vel.x, vel.y, vel.z])]
            lane_offset, lane_angle, pos_xy = self.get_lane_features()
            goal_features = self._get_goal_features()

        return {
            "lidar": self.lidar_data,
            "collision": np.array([self.collision_intensity], dtype=np.float32),
            "speed": np.array(speed, dtype=np.float32),
            "position": np.array(pos_xy, dtype=np.float32),
            "lane_offset": np.array([lane_offset], dtype=np.float32),
            "lane_angle": np.array([lane_angle], dtype=np.float32),
            "goal_direction": goal_features[:2],  # dx, dy normalisés vers l'objectif
            "goal_distance": np.array([goal_features[2]], dtype=np.float32),  # distance à l'objectif
        }

    # ----------------------------
    # Environnement Gym : reset, step, close
    # ----------------------------
    def reset(self):
        # détruit acteurs précédents
        self._destroy_all()
        time.sleep(0.05)

        # Spawn véhicule
        try:
            blueprints = self.world.get_blueprint_library()
            tesla = blueprints.find('vehicle.tesla.model3')
            spawn_points = self.world.get_map().get_spawn_points()
            spawn = random.choice(spawn_points) if spawn_points else carla.Transform(carla.Location(x=0,y=0,z=0))
            self.vehicle = self.world.spawn_actor(tesla, spawn)
            self.spawn_location = spawn.location
        except Exception as e:
            print("[CARLA] Erreur spawn véhicule:", e)
            raise

        # Attacher capteurs
        self._attach_collision_sensor()
        self._attach_lidar()

        # Parametrage destination final
        self._set_random_destination(min_dist=50.0, max_dist=50.0)

        # Reset flag collision critique
        self.critical_collision = False
        
        # Reset variables de tracking
        self.previous_distance_to_goal = None
        self.previous_position = None
        self.stationary_steps = 0
        self.max_distance_from_spawn = 0.0
        
        # Reset variables de lane keeping
        self.lane_keeping_streak = 0
        self.off_road_steps = 0
        
        # reset états internes
        self.collision_intensity = 0.0
        self.lidar_data = np.zeros((self.lidar_bins,), dtype=np.float32)

        # Si monde synchrone, avancer d'un tick pour que les capteurs fournissent des données
        try:
            if self.sync:
                self.world.tick()
            else:
                self.world.wait_for_tick(timeout=2.0)
        except Exception:
            time.sleep(0.1)

        self.episode_start_time = time.time()
        return self._get_obs()


# Notion évalué
# - Distance parcourue
# - Collisions
# - Lane angle
# - Lane offset

    def reward_function(self, obs):
        elapsed = time.time() - self.episode_start_time 
        collision = float(self.collision_intensity)
        vehicle_loc = self.vehicle.get_transform().location
        current_speed = obs["speed"][0]
        
        # Récupération des valeurs de lane keeping
        lane_offset = abs(obs["lane_offset"][0])  # Valeur absolue pour la distance
        lane_angle = abs(obs["lane_angle"][0])    # Valeur absolue pour l'angle
        
        # Calcul de la distance du spawn
        distance_from_spawn = vehicle_loc.distance(self.spawn_location) if self.spawn_location else 0.0
        self.max_distance_from_spawn = max(self.max_distance_from_spawn, distance_from_spawn)
        
        # Détection d'immobilité
        if self.previous_position is not None:
            movement = vehicle_loc.distance(carla.Location(x=self.previous_position[0], y=self.previous_position[1], z=0))
            if movement < 0.02:  
                self.stationary_steps += 1
            else:
                self.stationary_steps = 0
        
        # Mise à jour des positions précédentes
        self.previous_position = [vehicle_loc.x, vehicle_loc.y]
        
        # === ÉVALUATION DU LANE KEEPING ===
        # Seuils pour définir une bonne conduite
        excellent_offset_threshold = 0.1   # Très centré sur la voie
        good_offset_threshold = 0.3        # Bien centré
        bad_offset_threshold = 0.7         # Limite acceptable
        critical_offset_threshold = 0.9    # Presque hors route
        
        excellent_angle_threshold = 0.05   # Très bien aligné 
        good_angle_threshold = 0.15        # Bien aligné
        bad_angle_threshold = 0.4          # Limite acceptable
        critical_angle_threshold = 0.7     # Presque perpendiculaire
        
        # Tracking du comportement de conduite
        is_excellent_keeping = (lane_offset <= excellent_offset_threshold and 
                               lane_angle <= excellent_angle_threshold)
        is_good_keeping = (lane_offset <= good_offset_threshold and 
                          lane_angle <= good_angle_threshold)
        is_off_road = (lane_offset >= critical_offset_threshold or 
                      lane_angle >= critical_angle_threshold)
        
        # Mise à jour des compteurs
        if is_excellent_keeping or is_good_keeping:
            self.lane_keeping_streak += 1
            self.off_road_steps = 0
        else:
            self.lane_keeping_streak = max(0, self.lane_keeping_streak - 2)  # Décroissance rapide
            if is_off_road:
                self.off_road_steps += 1
            else:
                self.off_road_steps = 0
        
        # === CONDITIONS DE FIN D'ÉPISODE ===
        done = (
            collision > 50.0 or
            elapsed >= 500.0 or
            self.off_road_steps > 30 or  # Trop longtemps hors route
            lane_offset > 0.95           # Complètement hors de la voie
        )
        
        # === SYSTÈME DE RÉCOMPENSES AMÉLIORÉ ===
        
        # 1. RÉCOMPENSE DE BASE
        base_reward = 0.01
        
        # 2. RÉCOMPENSE DE LANE KEEPING PROGRESSIVE
        if is_excellent_keeping:
            # Excellente conduite : récompense forte et croissante
            lane_reward = 0.25 + min(self.lane_keeping_streak * 0.02, 0.3)
        elif is_good_keeping:
            # Bonne conduite : récompense modérée
            lane_reward = 0.1 + min(self.lane_keeping_streak * 0.01, 0.1)
        elif lane_offset <= bad_offset_threshold and lane_angle <= bad_angle_threshold:
            # Conduite acceptable : récompense faible
            lane_reward = 0.025
        else:
            # Mauvaise conduite : pénalité progressive
            offset_penalty = -2.0 * (lane_offset ** 2)  # Pénalité quadratique
            angle_penalty = -3.0 * (lane_angle ** 2)    # Pénalité quadratique plus forte
            lane_reward = offset_penalty + angle_penalty
        
        # 3. BONUS POUR CONDUITE SOUTENUE
        consistency_bonus = 0.0
        if self.lane_keeping_streak >= 50:
            consistency_bonus = 0.2
        elif self.lane_keeping_streak >= 20:
            consistency_bonus = 0.1
        elif self.lane_keeping_streak >= 10:
            consistency_bonus = 0.05
        
        # # 4. RÉCOMPENSE DE VITESSE (conditionnelle au lane keeping)
        if is_good_keeping or is_excellent_keeping:
            # Récompenser la vitesse seulement si bien centré
            speed_reward = 0.1 * np.clip(current_speed / 10.0, 0.0, 1.0)
        else:
            # Pénaliser la vitesse si mal centré
            speed_reward = -0.05 * np.clip(current_speed / 10.0, 0.0, 1.0)
        
        # 5. RÉCOMPENSE D'EXPLORATION
        exploration_reward = min(distance_from_spawn * 0.05, 0.1)
        
        # 6. PÉNALITÉS CRITIQUES
        collision_penalty = -100.0 if collision > 0.0 else 0.0
        immobility_penalty = -0.002 * max(self.stationary_steps, 0)
        off_road_penalty = -0.1 * (self.off_road_steps ** 1.5)  # Pénalité croissante
        off_road_termination_penalty = -50.0 if (self.off_road_steps > 30 or lane_offset > 0.95) else 0.0
        
        # === ASSEMBLAGE FINAL ===
        reward_components = {
            "base": base_reward,
            "lane_keeping": lane_reward,
            "consistency_bonus": consistency_bonus,
            "speed": speed_reward,
            "exploration": exploration_reward,
            "collision": collision_penalty,
            "immobility": immobility_penalty,
            "off_road": off_road_penalty,
            "off_road_termination": off_road_termination_penalty,
        }
        
        total_reward = sum(reward_components.values())
        
        # Clipping de sécurité pour éviter les récompenses extrêmes
        total_reward = np.clip(total_reward, -150.0, 10.0)
        
        return total_reward, done, reward_components

    def step(self, action):
        """
        action: np.array([throttle, brake, steer]) 
        """
        # appliquer contrôles
        action = action.squeeze(0) 
        steer = float(action[0].item())
        throttle = float(action[1].item())
        brake = float(action[2].item())
        reverse = False

        control = carla.VehicleControl()
        
        # Forcer un throttle minimum pour éviter l'immobilité
        min_throttle = 0.5
        control.throttle = np.clip(throttle, min_throttle, 1.0)
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.brake = np.clip(brake, 0.0, 1.0)
        control.reverse = False
        control.hand_brake = False
        control.manual_gear_shift = False

        # Logique pour éviter l'immobilité : si vitesse très faible, forcer throttle
        if self.vehicle:
            current_velocity = self.vehicle.get_velocity()
            current_speed = np.linalg.norm([current_velocity.x, current_velocity.y, current_velocity.z])
            
            if current_speed < 1.0:  # Si vitesse < 1 m/s
                control.throttle = 0.7  # Forcer un throttle plus élevé
                control.brake = 0.0

        if self.vehicle:
            try:
                self.vehicle.apply_control(control)
            except Exception:
                pass

        # Avancer la simulation (tick si synchrone)
        try:
            if self.sync:
                self.world.tick()
            else:
                # attendre une boucle physique
                self.world.wait_for_tick(timeout=1.0)
        except Exception:
            time.sleep(0.05)

        # mettre à jour spectateur
        self._vehicule_camera()

        # prélever observations
        obs = self._get_obs()

        # sécurité si pas de vehicle
        if self.vehicle is None:
            return obs, -10.0, True

        # calculer récompense
        reward, done, reward_coponents = self.reward_function(obs)

        # small shaping for staying alive (optional)
        return obs, float(reward), bool(done), reward_coponents

    # ----------------------------
    # Spawn / destroy helpers
    # ----------------------------
    def _destroy_all(self):
        actors = [self.collision_sensor, self.lidar_sensor, self.vehicle]
        for a in actors:
            if a is not None:
                try:
                    a.stop() if hasattr(a, "stop") else None
                except Exception:
                    pass
                try:
                    a.destroy()
                except Exception:
                    pass

        self.cam_sensor = None
        self.collision_sensor = None
        self.lidar_sensor = None
        self.position_sensor = None
        self.vehicle = None

        # Libérer la mémoire GPU
        torch.cuda.empty_cache()
        gc.collect()

    def close(self):
        self._destroy_all()
        print("[CARLA] Environnement fermé.")


