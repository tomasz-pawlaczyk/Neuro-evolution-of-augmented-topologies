import gymnasium as gym
import neat
import pickle
from gymnasium.wrappers import TimeLimit

ENV_NAME = "CartPole-v1"


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make(ENV_NAME)

    obs, _ = env.reset()
    fitness = 0
    done = False

    while not done:
        action = 1 if net.activate(obs)[0] > 0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        fitness += reward

    env.close()
    return fitness


def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-neat.txt"
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes, 50)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\n✅ The best-performing individual was saved to winner.pkl")


if __name__ == "__main__":
    run()
