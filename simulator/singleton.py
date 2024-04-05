import sys
import simpy


class ArgumentSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if ArgumentSingleton._instance is None:
            ArgumentSingleton._instance = ArgumentSingleton()
        return ArgumentSingleton._instance

    def __init__(self):
        if len(sys.argv) == 5:
            self.nodes_count = int(sys.argv[1])
            self.data_size = int(sys.argv[2])
            self.avg_wake_up_time = int(sys.argv[3])
            self.sim_time = int(sys.argv[4])

        else:
            print(
                "usage: ./main <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>"
            )
            exit(-1)


class DataGatewaySingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if DataGatewaySingleton._instance is None:
            DataGatewaySingleton._instance = DataGatewaySingleton()
        return DataGatewaySingleton._instance

    @staticmethod
    def reset():
        if DataGatewaySingleton._instance is not None:
            # Reset the internal state of the singleton instance as necessary
            # For example, re-initialize the data_gateway attribute
            from simulator.entities import DataGateway

            DataGatewaySingleton._instance.data_gateway = DataGateway(-1)
            # Reset any other necessary state here

    def __init__(self, data_id=-1):
        from simulator.entities import DataGateway

        self.data_gateway = DataGateway(data_id)
