import gymnasium as gym
import neat
import pickle
import time

ENV_NAME = "CartPole-v1"


def run_winner():
    with open("winner.pkl", "rb") as f:
        winner = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-neat.txt"
    )

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    env = gym.make(ENV_NAME, render_mode="human")
    obs, _ = env.reset()

    done = False
    while not done:
        action = 1 if net.activate(obs)[0] > 0 else 0
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    run_winner()
