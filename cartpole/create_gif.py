import gymnasium as gym
import neat
import pickle
import imageio
import numpy as np

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

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    obs, _ = env.reset()

    frames = []
    done = False

    while not done:
        action = 1 if net.activate(obs)[0] > 0 else 0
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = env.render()
        frames.append(frame)

    env.close()
    imageio.mimsave("cartpole.gif", frames, fps=30)


if __name__ == "__main__":
    run_winner()
