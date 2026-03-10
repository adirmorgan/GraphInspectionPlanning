
def pick_exp(Experiment):
    if Experiment == "Crisp1000":
        vertex_file = "./benchmarks/crisp/crisp_n1000_g1_vertex"
        edge_file = "./benchmarks/crisp/crisp_n1000_g1_edge"
        conf_file = "./benchmarks/crisp/crisp_n1000_g1_conf"
    elif Experiment == "Crisp2000":
        vertex_file = "./benchmarks/crisp/crisp_n2000_g1_vertex"
        edge_file = "./benchmarks/crisp/crisp_n2000_g1_edge"
        conf_file = "./benchmarks/crisp/crisp_n2000_g1_conf"
    elif Experiment == "Bridge1000":
        vertex_file = "./benchmarks/bridge/bridge_n1000_g1_vertex"
        edge_file = "./benchmarks/bridge/bridge_n1000_g1_edge"
        conf_file = "./benchmarks/bridge/bridge_n1000_g1_conf"
    elif Experiment == "Bridge2000":
        vertex_file = "./benchmarks/bridge/bridge_n2000_g1_vertex"
        edge_file = "./benchmarks/bridge/bridge_n2000_g1_edge"
        conf_file = "./benchmarks/bridge/bridge_n2000_g1_conf"

    return vertex_file, edge_file, conf_file
