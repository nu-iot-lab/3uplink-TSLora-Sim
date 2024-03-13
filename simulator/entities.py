import simulator.consts as consts
import numpy as np
from simulator.singleton import EnvironmentSingleton, DataGatewaySingleton, ArgumentSingleton
import random, math
from simulator.utils import *
from simulator.communications import DataPacket
from simulator.broadcast_traffic import BroadcastTraffic
from simulator.frame import Frame

environment = EnvironmentSingleton.get_instance()
args = ArgumentSingleton.get_instance()


class NetworkNode:
    def __init__(self, node_id=None):
        if node_id is not None:
            self.node_id = node_id
        self.x, self.y = 0, 0


class Gateway(NetworkNode):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        self.x, self.y = consts.bsx, consts.bsy

    @staticmethod
    def is_gateway():
        return True

    def __str__(self):
        return "gateway"


class DataGateway(Gateway):
    def __init__(self, node_id):
        super().__init__(node_id)
        self.frames = [Frame(sf) for sf in range(7, 10)]

    def frame(self, sf):
        if sf > 6:
            return self.frames[sf - 7]
        raise ValueError("sf must be greater than 6")

    def transmit_sack(self, env, sf):
        from simulator.communications import SackPacket

        # main sack packet transmission loop
        while True:
            yield env.timeout(self.frame(sf).trans_time - env.now)
            sack_packet = SackPacket(self.frame(sf).nr_slots_SACK, sf, self)

            # print("-" * 70)
            if self.frame(sf).nr_taken_slots != 0:
                log(
                    env,
                    f"[SACK-TRANSMIT]"
                    f'{f"SF: {sf} ":<10}'
                    f'{f"Data size: {sack_packet.pl} b ":<20}'
                    f'{"":<25}'
                    f'{f"Freq: {sack_packet.freq / 1000000.0:.3f} MHZ ":<24}'
                    f'{f"BW: {sack_packet.bw}  kHz ":<18}'
                    f'{f"Airtime: {sack_packet.rec_time / 1000.0:.3f} s ":<22}',
                )
            # print("-" * 70)

            for n in consts.nodes:
                if n.connected and n.sf == sf:
                    env.process(self.transmit_sack_to_node(env, n, sf))

            yield env.timeout(self.frame(sf).next_round_start_time - env.now)
            self.frame(sf).next_frame()

    def transmit_sack_to_node(self, env, node, sf):
        from simulator.communications import SackPacket
        from simulator.singleton import DataGatewaySingleton

        data_gateway = DataGatewaySingleton.get_instance().data_gateway
        sack_packet = SackPacket(self.frame(sf).nr_slots_SACK, sf, self)
        sack_packet.add_time = env.now

        if sack_packet.is_lost(node):
            sack_packet.lost = True
            log(
                env,
                f"[SACK-FAIL] {self} transmit to {node} SACK failed, too much path loss: {sack_packet.rssi(node)}",
            )

        sack_packet.check_collision()

        yield BroadcastTraffic.add_and_wait(env, sack_packet)
        sack_packet.update_statistics()
        data_gateway.frame(sf).check_data_collision()

        if sack_packet.is_received():
            log(env, f"[SACK-RECEIVED] {node} received SACK packet")
            node.round_start_time = self.frame(sf).next_round_start_time
            node.network_size = self.frame(sf).nr_slots
            node.guard_time = self.frame(sf).guard_time
            node.frame_length = self.frame(sf).frame_length
            if node.waiting_first_sack:
                node.sack_packet_received.succeed()
        else:
            log(env, f"[SACK-NOT-RECEIVED] Sack packet was not received by {node}")
        sack_packet.reset()


class EndNode(NetworkNode):
    def __init__(self, node_id, gateway=None):
        super().__init__(node_id)
        self.data_gateway = DataGatewaySingleton.get_instance().data_gateway
        self.missed_sack_count = 0
        self.packets_sent_count = 0
        self.packets_received_count = 0 

        self.prr_value = 0
        self.rssi_value = 0
        self.sf_value = 0

        self.state = [self.prr_value, self.rssi_value, self.sf_value]

        #  consts.nr_sack_missed_count + consts.nr_lost + consts.nr_data_collisions = consts.nr_data_retransmissions
        self.nr_lost = 0
        self.nr_data_collisions = 0

        self.connected = True  # for now
        self.accept_received = False
        self.waiting_first_sack = True

        self.round_start_time = 0
        self.round_end_time = 0

        # for triple uplink
        self.slot = [None, None, None]
        self.counter_index = 0
        self.guard_time = 2
        self.frame_length = 0
        self.network_size = 0

        self.req_packet, self.data_packet = None, None
        self.sack_packet_received = environment.event()

        self.x, self.y = EndNode.find_place_for_new_node()
        self.dist = np.sqrt(
            (self.x - consts.bsx) * (self.x - consts.bsx)
            + (self.y - consts.bsy) * (self.y - consts.bsy)
        )


        self.sf = self.find_optimal_sf()
        print(
            f"node {self.node_id}: \t x {self.x:3f} \t y {self.y:3f} \t dist {self.dist:4.3f} \t SF {self.sf}"
        )

    def __str__(self):
        # return "EndNode: " + str(self.node_id) + " x: " + str(self.x) + " y: " + str(self.y) + " sf: " + str(self.sf)
        return f"EndNode: {self.node_id} sf: {self.sf}"

    @staticmethod
    def is_gateway():
        return False

    @staticmethod
    def find_place_for_new_node():
        found = False
        rounds = 0
        while not found and rounds < 100:
            a = random.random()
            b = random.random()

            if b < a:
                a, b = b, a
            posx = b * consts.max_dist * math.cos(2 * math.pi * a / b) + consts.bsx
            posy = b * consts.max_dist * math.sin(2 * math.pi * a / b) + consts.bsy

            if len(consts.nodes) == 0:
                found = True
                break

            for index, n in enumerate(consts.nodes):
                dist = np.sqrt(((abs(n.x - posx)) ** 2) + ((abs(n.y - posy)) ** 2))
                if dist >= 10:
                    found = True
                else:
                    found = False
                    rounds += 1
                    if rounds == 100:
                        print("could not find place for a new node, giving up")
                        exit(-1)
        return posx, posy

    def find_optimal_sf(self):
        for sf in range(7, 10):
            for i in range(10):
                isLost = False
                data_packet = DataPacket(sf, self)
                if data_packet.rssi(self.data_gateway) < get_sensitivity(
                    sf, data_packet.bw
                ):
                    isLost = True
                    break
            if not isLost:
                return sf
        print(f"WARNING: {self} cannot reach gateway!")
        return None
    
    def calculate_prr(self):
        if self.packets_sent_count > 0:
            return self.packets_received_count / self.packets_sent_count
        else:
            return 0

            
    def update_state(self):
        """Updates the node's state based on current PRR, RSSI, and SF."""
        self.state = [self.calculate_prr(), self.rssi_value, self.sf]

    def select_action_based_on_state(self):
        """Determines the number of retransmissions based on the node's state."""
        # Placeholder for your DQL model's action prediction
        # This example uses a simplistic rule-based approach for illustration
        if self.state[0] > 0.98 and self.state[1] > -120:  # High PRR and good RSSI
            return 0  # Fewer retransmissions needed
        elif self.state[2] <= 7:  # Low SF, indicating closer proximity
            return 1  # Possibly fewer retransmissions
        else:
            return 2  # Default or more retransmissions for lower PRR or worse RSSI

    def perform_action(self, action):
        """Executes the chosen action."""
        # Adjust the node's behavior based on the selected action, e.g., adjust retransmission strategy
        self.max_retransmissions = action

    def transmit(self, env):
        while True:
            # calculating round start time
            yield env.timeout(random.uniform(0.0, float(2 * args.avg_wake_up_time)))
            if self.waiting_first_sack:
                yield self.sack_packet_received
                # print(f"Node {self.node_id} received its first SACK packet at simulation time {env.now}.")
                self.waiting_first_sack = False
                self.sack_packet_received = environment.event()
            else:
                # env.timeout(self.round_end_time - env.now)
                yield env.timeout(self.round_end_time - env.now)

            if self.round_start_time < env.now:
                log(env, f"[SACK-MISSED] {self}: missed sack packet")
                self.round_start_time = env.now + 1
                self.missed_sack_count += 1
                consts.nr_sack_missed_count += 1
            else:
                self.missed_sack_count = 0

            # reconnecting to gateway if too many SACK-s missed
            # if self.missed_sack_count == 3:
            #     log(env, "[NODE-RECONNECTION] node {}: reconnecting to the gateway. ".format(self.node_id))
            #     # self.connected = False
            #     data_gateway.frame(self.sf).remove(self)
            #     continue
            # print(f"Node {self.node_id} waiting {max(0, self.round_start_time - env.now)} until next transmission opportunity.")
            yield env.timeout(self.round_start_time - env.now)

            # calculating round_end_time and waiting till send_time
            self.round_end_time = env.now + self.frame_length
            send_time = (self.slot[0] + self.counter_index) * (
                DataPacket(self.sf).rec_time + 2 * self.guard_time
            ) + self.guard_time
            self.counter_index += 1
            if self.counter_index == 2:
                self.counter_index = 0
            if self.slot[0] != 0:
                send_time = send_time + random.gauss(0, consts.sigma) * self.guard_time
            yield env.timeout(send_time)


            self.update_state()
            action = self.select_action_based_on_state()
            self.perform_action(action)

            data_packet = DataPacket(self.sf, self)
            data_packet.add_time = env.now
            data_packet.sent = True

            if (data_packet.sent): 
                self.packets_sent_count += 1 

            if not data_packet.lost and data_packet.rssi(self.data_gateway) >= get_sensitivity(data_packet.sf, data_packet.bw):
                # Packet is considered successfully received if not lost and RSSI is above sensitivity
                self.packets_received_count += 1 

            prr_value = self.calculate_prr()
            rssi_value = data_packet.rssi(self.data_gateway)
            sf_value = data_packet.sf
            # [NODE-SEND-PACKET] -- node {self.node_id} sent data packet

            # values for logging
            node_send_packet = f"[NODE-SEND-PACKET-{self.node_id}]"
            data_size = f"Data size: {data_packet.pl} b"
            freq = f"Freq: {data_packet.freq / 1000000.0:.3f} MHZ"
            bw = f"BW: {data_packet.bw} kHz"
            airtime = f"Airtime: {data_packet.rec_time / 1000.0:.3f} s"
            guardtime = f"Guardtime: {self.guard_time / 1000.0:.3f} ms"
            rssi = f"RSSI: {rssi_value:.3f} dBm"
            prr = f"PRR: {prr_value:.3f}"
            sf = f"SF: {sf_value}"

            # logging the message
            log_message = f"{node_send_packet:<40}{data_size:<20}{freq:<24}{bw:<18}{airtime:<22}{guardtime}\n{rssi:<25}{prr}\n{sf:<10}"
            log(env, log_message)


            if data_packet.rssi(self.data_gateway) < get_sensitivity(
                data_packet.sf, data_packet.bw
            ):
                log(env, f"[NODE-LOST] {self}: packet will be lost")
                data_packet.lost = True



            data_packet.check_collision()
            yield BroadcastTraffic.add_and_wait(env, data_packet)
            data_packet.update_statistics()
            data_packet.reset()
            
