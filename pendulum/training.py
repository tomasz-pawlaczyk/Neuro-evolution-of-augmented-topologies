import gymnasium as gym
import pickle
import os
import visualize
import neat
import multiprocessing
import numpy as np
import pandas as pd
from typing import List, Tuple, Any


runs_per_net = 4
max_steps = 200

N_POPULATIONS   = [100, 150, 200, 250]
CONN_PROB      = [0.05, 0.1, 0.2, 0.4, 0.6]
NODE_PROB      = [0.05, 0.1, 0.2, 0.3]
COMP_THRESHOLDS = [2.0, 3.0, 4.0]
ACTIVATIONS     = ['tanh', 'relu', 'sigmoid']


def eval_genome(genome: neat.DefaultGenome, config: neat.Config) -> float:
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []
    for _ in range(runs_per_net):
        env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
        observation, info = env.reset()
        fitness = 0.0
        for i in range(max_steps):
            output = net.activate(observation)
            action = [output[0] * 2.0]  
            
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            if terminated or truncated:
                break
        env.close()
        fitnesses.append(fitness)
    return sum(fitnesses) / len(fitnesses)

def eval_genomes(genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome , config)

def run(config_file: str) -> Tuple[neat.DefaultGenome, neat.StatisticsReporter]:
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, 1000)

    with open('winner- feedforward.pickle', 'wb') as f:
        pickle.dump(winner, f)
    visualize.plot_stats(stats, ylog=False, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {
        -1: 'cos(theta)', 
        -2: 'sin(theta)', 
        -3: 'theta',
        0: 'torque'
    }
    print(f'\n\nBest genome:\n{winner!s}')
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                      filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                      filename="winner-feedforward-pruned.gv", prune_unused=True)

    return winner, stats


def run_experiment_stage(config: neat.Config, param_name: str, values: List[Any], attr_path: str, is_population_stage: bool = False) -> Tuple[pd.DataFrame, Any]:
    results = []
    print(f"\n>>> ETAP: {param_name}")
    
    for val in values:
        obj = config
        parts = attr_path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)
        
        gen_counts = []
        cost_counts = []
        print(f"  Testuję {val}: ", end="", flush=True)
        
        for _ in range(10): 
            pop = neat.Population(config)
            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
            pop.run(pe.evaluate, 200) 
            
            gens = len(pop.reporters.reporters[0].most_fit_genomes) if hasattr(pop.reporters.reporters[0], 'most_fit_genomes') else 200
            gen_counts.append(gens)
            if is_population_stage:
                cost_counts.append(gens * val)
            print(".", end="", flush=True)
        
        avg_gen = np.mean(gen_counts)
        if is_population_stage:
            avg_cost = np.mean(cost_counts)
            row = {param_name: val, "Srednia_Kosztu": avg_cost, "Srednia_Gen": avg_gen}
        else:
            row = {param_name: val, "Srednia_Gen": avg_gen}
            
        results.append(row)
        print(f" Wynik: {avg_gen} gen")
    
    df = pd.DataFrame(results)
    best_val = df.loc[df['Srednia_Kosztu' if is_population_stage else 'Srednia_Gen'].idxmin(), param_name]
    return df, best_val

def test_parameters(config: neat.Config, N_POPULATIONS: List[int], CONN_PROBS: List[float], NODE_PROBS: List[float], COMP_THRESHOLDS: List[float], ACTIVATIONS: List[str]) -> None:
    all_reports = {}
    try:
        # etap 1
        df_pop, best_pop = run_experiment_stage(config, "pop_size", N_POPULATIONS, "pop_size", is_population_stage=True)
        config.pop_size = int(best_pop)
        all_reports["1_Populacja"] = df_pop

        # ETAP 2
        df_conn, best_conn = run_experiment_stage(config, "conn_add_prob", CONN_PROBS, "genome_config.conn_add_prob")
        config.genome_config.conn_add_prob = best_conn
        all_reports["2_Conn_Add_Prob"] = df_conn

        # ETAP 3
        df_node, best_node = run_experiment_stage(config, "node_add_prob", NODE_PROBS, "genome_config.node_add_prob")
        config.genome_config.node_add_prob = best_node
        all_reports["3_Node_Add_Prob"] = df_node

        # ETAP 4
        df_comp, best_comp = run_experiment_stage(config, "compatibility_threshold", COMP_THRESHOLDS, "species_set_config.compatibility_threshold")
        config.species_set_config.compatibility_threshold = best_comp
        all_reports["4_Compatibility"] = df_comp

        # ETAP 5
        config.genome_config.activation_mutate_rate = 0.0
        df_act, best_act = run_experiment_stage(config, "activation", ACTIVATIONS, "genome_config.activation_default")
        all_reports["5_Activation"] = df_act

    except Exception as e:
        print(f"errorinho gaucho {e}")


    print("\n" + "="*70)
    print("RAPORT Z OPTYMALIZACJI")
    print("="*70)
    for stage, report in all_reports.items():
            print(f"\n>>> Wyniki etapu: {stage}")
            print(report.to_string(index=False))



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    genome_path = os.path.join(local_dir, 'winner- feedforward.pickle')
    winner, stats = run(config_path)

