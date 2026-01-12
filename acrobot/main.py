import neat
import gymnasium as gym
import numpy as np
import pickle
from graph_visual import draw_net

def eval_genomes(genomes, config):
    env = gym.make("Acrobot-v1")

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        observation, info = env.reset()

        genome.fitness = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            output = net.activate(observation)

            action = np.argmax(output)

            observation, reward, done, truncated, info = env.step(action)

            genome.fitness += reward

    env.close()

def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 300)

    print('\nNajlepszy genom:\n{!s}'.format(winner))

    with open("winner-acrobot.pkl", "wb") as f:
        pickle.dump(winner, f)

    return winner, config


def replay_genome(genome, config):
    env = gym.make("Acrobot-v1", render_mode="human")
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    observation, info = env.reset()
    done = False
    truncated = False

    print("Odtwarzanie najlepszego modelu...")
    while not (done or truncated):
        output = net.activate(observation)
        action = np.argmax(output)
        observation, reward, done, truncated, info = env.step(action)

    env.close()


winner, config = run(config_path='config-acrobot')
draw_net(config,winner,filename="winner_graph.gv")
replay_genome(winner, config)