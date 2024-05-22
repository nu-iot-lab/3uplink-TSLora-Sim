import sys
import simpy


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
