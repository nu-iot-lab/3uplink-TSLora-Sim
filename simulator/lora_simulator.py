import simulator.consts as consts

from simulator.entities import EndNode
from simulator.singleton import DataGatewaySingleton

data_gateway = DataGatewaySingleton.get_instance().data_gateway


class LoraSimulator:
    def __init__(self, nodes_count, data_size, avg_wake_up_time, sim_time, env):
        self.nodes_count = nodes_count
        self.data_size = data_size
        self.avg_wake_up_time = avg_wake_up_time
        self.sim_time = sim_time
        self.env = env

    def add_nodes(self):
        print("\n!--NODES--!\n")
        for i in range(self.nodes_count):
            consts.nodes.append(EndNode(i, self.env, data_gateway))

    def update_nodes_behavior(self, action):
        for node in consts.nodes:
            node.perform_action(action)

    def start_simulation(self):
        # Initialize slots inside the frame
        for sf in range(7, 10):
            data_gateway.frame(sf).assign_slots()

        for node in consts.nodes:
            self.env.process(node.transmit(self.env))

        for sf in range(7, 10):
            self.env.process(data_gateway.transmit_sack(self.env, sf))

    def __str__(self):
        return (
            "LoraSimulator(nodes_count=%s, data_size=%s, avg_wake_up_time=%s, sim_time=%s)"
            % (self.nodes_count, self.data_size, self.avg_wake_up_time, self.sim_time)
        )
