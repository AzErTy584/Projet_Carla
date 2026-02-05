import carla
import random
import os
import time
import keyboard  # Pour détecter les touches du clavier
import numpy as np

CARLA_HOST = 'localhost'
CARLA_PORT = 2000

# Lancer Carla avec :  CarlaUE4.exe -fps=20 -world-port=3000

# Connexion au serveur CARLA
def connect_to_carla(host=CARLA_HOST, port=CARLA_PORT):
    try:
        client = carla.Client(host, port)
        client.set_timeout(20.0)
        print(f"Connexion réussie au serveur CARLA sur {host}:{port} !")
        return client
    except Exception as e:
        print(f"Échec de la connexion au serveur CARLA : {e}")
        return None
# Création de l'environnement CARLA
def create_environment(client):
    try:
        world = client.get_world()
        print("Environnement CARLA chargé avec succès !")
        return world
    except Exception as e:
        print(f"Échec du chargement de l'environnement CARLA : {e}")
        return None

def spawn_actor(world):
    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.ford.mustang')  # Véhicule spécifique
        # vehicle_bp = random.choice(blueprint_library.filter('vehicle')) #Random vehicle
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            print(f"Véhicule ajouté avec succès : {vehicle.type_id}")
            return vehicle
        else:
            print("Échec de l'ajout du véhicule.")
            return None
    except Exception as e:
        print(f"Échec de l'ajout de l'acteur : {e}")
        return None
    
def spectate_actor(world, actor_id):
    try:
        spectator = world.get_spectator()
        actor = world.get_actor(actor_id)
        if actor:
            # Positionne le spectateur légèrement en arrière et en hauteur
            tf = actor.get_transform()
            offset = carla.Location(x=-5, z=3)  # Position du spectateur
            spectator_pos = tf.transform(offset)

            spectator.set_transform(carla.Transform(
                spectator_pos,
                tf.rotation
            ))
            time.sleep(0.01) # Petit délai pour stabiliser la vue



        else:
            print(f"Aucun acteur trouvé avec l'ID {actor_id}.")
    except Exception as e:
        print(f"Échec du positionnement du spectateur ou de l'attachement de la caméra : {e}")


def destroy_actor(actor):
    try:
        if actor:
            actor.destroy()
            print(f"Acteur {actor.id} détruit avec succès.")
        else:
            print("Aucun acteur à détruire.")
    except Exception as e:
        print(f"Échec de la destruction de l'acteur : {e}")


def spawn_new_vehicule(world,number=10):
    try:
        for i in range(number):  # Essayer plusieurs fois de spawn un véhicule
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = random.choice(blueprint_library.filter('vehicle')) #Random vehicle
            spawn_points = world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_autopilot(True)

        if vehicle:
            print(f"Véhicules ajoutés avec succès.")
            return vehicle
        else:
            print("Échec de l'ajout des véhicules.")
            return None
    except Exception as e:
        print(f"Échec de l'ajout des acteurs : {e}")
        return None


def control_vehicle_with_keyboard(vehicle):
    try:
        control = carla.VehicleControl()
        print("Contrôle du véhicule avec le clavier. Utilisez les touches suivantes :")
        print("- Flèche haut : Accélérer")
        print("- Flèche bas : Freiner/Reculer")
        print("- Flèche gauche : Tourner à gauche")
        print("- Flèche droite : Tourner à droite")
        print("- Espace : Arrêter le véhicule")
        print("- Échap : Quitter")

        while True:
            # Réinitialise les commandes
            control.throttle = 0.0
            control.brake = 0.0
            control.steer = 0.0
            control.reverse = False

            spectate_actor(vehicle.get_world(), vehicle.id)

            # Détecte les touches pressées
            if keyboard.is_pressed('up'):
                control.throttle = 0.7
            if keyboard.is_pressed('down'):
                control.brake = 0.5
                control.reverse = True
            if keyboard.is_pressed('left'):
                control.steer = -0.7
            if keyboard.is_pressed('right'):
                control.steer = 0.7
            if keyboard.is_pressed('space'):
                control.throttle = 0.0
                control.brake = 1.0
            if keyboard.is_pressed('shift'):
                spawn_new_vehicule(vehicle.get_world())
            if keyboard.is_pressed('esc'):
                print("Fin du contrôle.")
                destroy_actor(vehicle)
                break

            # Applique les commandes au véhicule
            vehicle.apply_control(control)

    except Exception as e:
        print(f"Erreur lors du contrôle du véhicule : {e}")
    

# Fonction principale
def main(world):
    try:
        vehicles = [actor for actor in world.get_actors() if 'vehicle' in actor.type_id]

        if not vehicles:
            vehicle = spawn_actor(world)
        else:
            vehicle = random.choice(vehicles)

        if vehicle:
            control_vehicle_with_keyboard(vehicle)

    except Exception as e:
        print(f"Erreur dans la fonction principale : {e}")


if __name__ == '__main__':
    client = connect_to_carla()
    if client:
        world = create_environment(client)
        if world:
            main(world)
    else:
        print("Impossible de se connecter au serveur CARLA.")
