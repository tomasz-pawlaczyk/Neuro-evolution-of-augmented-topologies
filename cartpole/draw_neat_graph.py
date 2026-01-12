import neat
import pickle
from graphviz import Digraph


def draw_net(config, genome, filename="neat_graph"):
    dot = Digraph(format="png")
    dot.attr(rankdir="LR", size="8,5")

    for node_id, node in genome.nodes.items():
        if node_id < 0:
            dot.node(str(node_id), f"Input {abs(node_id)}", shape="box")
        elif node_id in config.genome_config.output_keys:
            dot.node(str(node_id), f"Output {node_id}", shape="doublecircle")
        else:
            dot.node(str(node_id), f"Hidden {node_id}", shape="circle")

    for conn_key, conn in genome.connections.items():
        if not conn.enabled:
            continue

        color = "green" if conn.weight > 0 else "red"
        label = f"{conn.weight:.2f}"

        dot.edge(
            str(conn_key[0]),
            str(conn_key[1]),
            label=label,
            color=color
        )

    dot.render(filename, cleanup=True)
    print(f"✅ Graf zapisany do {filename}.png")


def main():
    with open("winner.pkl", "rb") as f:
        genome = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-neat.txt"
    )

    draw_net(config, genome)


if __name__ == "__main__":
    main()
