import sys
import simpy


class EnvironmentSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if EnvironmentSingleton._instance is None:
            EnvironmentSingleton._instance = simpy.Environment()
        return EnvironmentSingleton._instance


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
            print("usage: ./main <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>")
            exit(-1)


class DataGatewaySingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if DataGatewaySingleton._instance is None:
            DataGatewaySingleton._instance = DataGatewaySingleton()
        return DataGatewaySingleton._instance

    def __init__(self, data_id=-1):
        from entities import DataGateway
        self.data_gateway = DataGateway(data_id)
