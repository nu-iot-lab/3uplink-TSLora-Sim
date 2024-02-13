# from entities import NetworkNode
from broadcast_traffic import BroadcastTraffic
from utils import *
from consts import *
import math
import random
from channels import Channels
from singleton import ArgumentSingleton


class Packet:
    def __init__(self, node=None, receiver=None):
        from entities import NetworkNode

        self.node = NetworkNode() if node is None else node
        self.receiver = None if receiver is None else receiver

        self.cr = coding_rate
        self.bw, self.sf, self.pl, self.rec_time = 0, 0, 0, 0

        self.collided = False
        self.processed = False
        self.lost = False
        self.sent = False
        self.add_time = None

    def energy_transmit(self):
        return self.airtime() * (pow_cons[0] + pow_cons[2]) * V / 1e6

    def energy_receive(self):
        if self.is_received():
            return (50 + self.airtime()) * (pow_cons[1] + pow_cons[2]) * V / 1e6
        return 0

    def dist(self, destination):
        return np.sqrt(
            (self.node.x - destination.x) * (self.node.x - destination.x)
            + (self.node.y - destination.y) * (self.node.y - destination.y)
        )

    def rssi(self, destination):
        # xs = variance * random.gauss(0, 0.01)
        # + np.random.normal(-variance, variance)
        Lpl = (
            Lpld0
            + 10 * gamma * math.log10(self.dist(destination) / d0)
            + np.random.normal(-variance, variance)
        )
        Prx = Ptx - GL - Lpl
        return Prx  # threshold is 12 dB

    def is_lost(self, destination):
        rssi = self.rssi(destination)
        sens = get_sensitivity(self.sf, self.bw)
        return rssi < sens

        # this function computes the airtime of a packet according to LoraDesignGuide_STD.pdf

    def airtime(self):
        H = 0  # implicit header disabled (H=0) or not (H=1)
        DE = 0  # low data rate optimization enabled (=1) or not (=0)
        Npream = 8  # number of preamble symbol (12.25  from Utz paper)

        if self.bw == 125 and self.sf in [11, 12]:
            DE = 1  # low data rate optimization mandated for BW125 with SF11 and SF12
        if self.sf == 6:
            H = 1  # can only have implicit header with SF6

        Tsym = (2.0**self.sf) / self.bw
        Tpream = (Npream + 4.25) * Tsym
        payloadSymbNB = 8 + max(
            math.ceil(
                (8.0 * self.pl - 4.0 * self.sf + 28 + 16 - 20 * H)
                / (4.0 * (self.sf - 2 * DE))
            )
            * (self.cr + 4),
            0,
        )
        Tpayload = payloadSymbNB * Tsym
        return Tpream + Tpayload

    def reset(self):
        self.collided = False
        self.processed = False
        self.lost = False

    def update_statistics(self):
        if self.lost:
            global nr_lost
            nr_lost += 1

        if self.collided:
            global nr_collisions
            nr_collisions += 1

        if self.is_received():
            global nr_received
            nr_received += 1

        if self.processed:
            global nr_processed
            nr_processed += 1

        if self.sent:
            global nr_packets_sent
            nr_packets_sent += 1

        global total_energy
        global erx
        global etx
        erx += self.energy_receive()
        etx += self.energy_transmit()
        total_energy += self.energy_transmit() + self.energy_receive()

    def is_received(self):
        return not self.collided and self.processed and not self.lost

    def was_sent_to(self, node):
        return self.receiver is node

    def check_collision(self):
        self.processed = True
        if BroadcastTraffic.nr_data_packets > max_packets:
            log(
                env, "[PACKET-OVERFLOW] too many packets are being sent to the gateway:"
            )
            self.processed = False

        if BroadcastTraffic.nr_packets:
            for other in BroadcastTraffic.traffic:
                if self.node is other.node:
                    continue

                if (
                    self.node.is_gateway() != other.node.is_gateway()
                    and self.sf == other.sf
                ):
                    if self.processed and self.was_sent_to(other.node):
                        log(env, f"[PACKET-DROP] {self} from {self.node} is dropped")
                        self.processed = False

                    if other.processed and other.was_sent_to(self.node):
                        log(
                            env,
                            f"[PACKET-DROP-OTHER] {other} from {other.node} is dropped",
                        )
                        other.processed = False

                if (
                    frequency_collision(self, other)
                    and sf_collision(self, other)
                    and timing_collision(self, other)
                ):
                    for p in power_collision(self, other):
                        p.collided = True
                        if p == self:
                            p2 = other
                        else:
                            p2 = self
                        log(env, f"[COLLISION] {p.node} collided with {p2.node}")


class DataPacket(Packet):
    def __init__(self, sf=None, node=None):
        super().__init__(node, data_gateway)
        if sf not in [7, 8, 9]:
            sf = random.choice([7, 8, 9])
        self.sf = sf
        self.bw = 125
        self.freq = Channels.get_sf_freq(sf)
        self.pl = ArgumentSingleton.get_instance().data_size
        self.rec_time = self.airtime()

    def update_statistics(self):
        super().update_statistics()
        if self.sent:
            global nr_data_packets_sent
            nr_data_packets_sent += 1

        if self.sent and self.node is not None:
            self.node.packets_sent_count += 1

    def __str__(self):
        return "data packet"


class SackPacket(Packet):
    def __init__(self, nr_slots, sf=None, node=None):
        super().__init__(node, None)
        self.sf = sf
        self.bw = 125
        self.freq = Channels.get_sf_freq(sf)
        self.pl = int(4 + (nr_slots + 7) / 8)
        self.rec_time = self.airtime()

    def update_statistics(self):
        super().update_statistics()
        if self.sent:
            global nr_sack_sent

    def was_sent_to(self, node):
        return self.sf == node.sf

    def __str__(self):
        return "SACK packet"
