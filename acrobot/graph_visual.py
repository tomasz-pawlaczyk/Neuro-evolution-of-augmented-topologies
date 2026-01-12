import graphviz


def draw_net(config, genome, view=False, filename=None):
    if filename is None:
        filename = 'winner_graph'

    node_names = {
        -1: 'cos(t1)', -2: 'sin(t1)',
        -3: 'cos(t2)', -4: 'sin(t2)',
        -5: 'vel(t1)', -6: 'vel(t2)',
        0: 'Lewo', 1: 'Nic', 2: 'Prawo'
    }

    dot = graphviz.Digraph(format='png')

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': 'lightgray'}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': 'lightblue'}
        dot.node(name, _attributes=node_attrs)

    for node in genome.nodes.values():
        if node.key not in inputs and node.key not in outputs:
            name = str(node.key)
            attrs = {'style': 'filled', 'fillcolor': 'white'}
            dot.node(name, _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled:
            input_node = node_names.get(cg.key[0], str(cg.key[0]))
            output_node = node_names.get(cg.key[1], str(cg.key[1]))

            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight))
            style = 'solid'
            dot.edge(input_node, output_node, _attributes={'color': color, 'penwidth': width, 'style': style})

    dot.render(filename, view=view)
    print(f"Graf sieci zapisano jako {filename}.png")
