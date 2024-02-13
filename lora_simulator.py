import consts
from entities import EndNode
from singleton import EnvironmentSingleton, DataGatewaySingleton


env = EnvironmentSingleton.get_instance()
data_gateway = DataGatewaySingleton.get_instance().data_gateway


class LoraSimulator:
    def __init__(self, nodes_count, data_size, avg_wake_up_time, sim_time):
        self.nodes_count = nodes_count
        self.data_size = data_size
        self.avg_wake_up_time = avg_wake_up_time
        self.sim_time = sim_time

    def add_nodes(self):
        print("\n!--NODES--!\n")
        for i in range(self.nodes_count):
            consts.nodes.append(EndNode(i, data_gateway))

    def start_simulation(self):
        print("\n!--START--!\n")
        # initialize slots inside the frame
        for sf in range(7, 10):
            data_gateway.frame(sf).assign_slots()

        for node in consts.nodes:
            env.process(node.transmit(env))

        for sf in range(7, 10):
            env.process(data_gateway.transmit_sack(env, sf))

        env.run(until=self.sim_time)
        print("\n!--END--!\n")

    def __str__(self):
        return (
            "LoraSimulator(nodes_count=%s, data_size=%s, avg_wake_up_time=%s, sim_time=%s)"
            % (self.nodes_count, self.data_size, self.avg_wake_up_time, self.sim_time)
        )
