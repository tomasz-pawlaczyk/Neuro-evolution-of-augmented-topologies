import os
import pickle
import gymnasium as gym
import neat
import imageio
import numpy as np
from typing import List, Any

def test_network(net: Any, episodes: int = 10, render: bool = True, camera_distance: float = 4.0) -> List[float]:
    fitnesses = []
    env = gym.make("Pendulum-v1", render_mode="human")
    
    try:
        for episode in range(episodes):
            observation, info = env.reset()
            
            if episode == 0 and render and hasattr(env.unwrapped, 'mujoco_renderer'):
                renderer = env.unwrapped.mujoco_renderer
                if renderer.viewer is not None:
                    renderer.viewer.cam.distance = camera_distance
            
            fitness = 0.0
            step = 0
            
            while True:
                step += 1
                action = net.activate(observation)
                
                observation, reward, terminated, truncated, info = env.step(action)
                fitness += reward
                
                if terminated or truncated:
                    break
            
            fitnesses.append(fitness)
            print(f"Episode {episode + 1}: steps={step}, fitness={fitness:.2f}")
    
    finally:
        env.close()
    
    avg_fitness = sum(fitnesses) / len(fitnesses)
    max_fitness = max(fitnesses)
    min_fitness = min(fitnesses)
    
    print(f"\nResults over {episodes} episodes:")
    print(f"  Average fitness: {avg_fitness:.2f}")
    print(f"  Max fitness: {max_fitness:.2f}")
    print(f"  Min fitness: {min_fitness:.2f}")
    
    return fitnesses

def save_gif(net: Any, filename: str = "pendulum_result.gif") -> None:
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    observation, _ = env.reset()
    frames = []
    for _ in range(200):
        frames.append(env.render())
        action = net.activate(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    imageio.mimsave(filename, [np.array(f) for f in frames], fps=30)

def load_and_test(genome_path: str, config_path: str, episodes: int = 10, render: bool = True, camera_distance: float = 4.0) -> List[float]:

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    print('Loaded genome:')
    print(genome)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    save_gif(net)
    
    return test_network(net, episodes=episodes, render=render, camera_distance=camera_distance)

if __name__ == '__main__':
    import sys

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    
    if len(sys.argv) > 1:
        genome_path = sys.argv[1]
    else:
        genome_path = os.path.join(local_dir, 'winner- feedforward.pickle')


    print(f"Testing genome from: {genome_path}\n")
    load_and_test(genome_path, config_path, episodes=5, render=True, camera_distance=4.0)