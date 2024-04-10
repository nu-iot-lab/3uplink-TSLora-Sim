from simulator.broadcast_traffic import BroadcastTraffic

from simulator.utils import *
import simulator.consts as consts
import numpy as np
import math
import random
from simulator.channels import Channels
from simulator.singleton import ArgumentSingleton


class Packet:
    def __init__(self, node=None, receiver=None):
        from simulator.entities import NetworkNode

        self.node = NetworkNode() if node is None else node
        self.receiver = None if receiver is None else receiver

        self.cr = consts.coding_rate
        self.bw, self.sf, self.pl, self.rec_time = 0, 0, 0, 0

        self.collided = False
        self.processed = False
        self.lost = False
        self.sent = False
        self.add_time = None

    def energy_transmit(self):
        return (
            self.airtime() * (consts.pow_cons[0] + consts.pow_cons[2]) * consts.V / 1e6
        )

    def energy_receive(self):
        if self.is_received():
            return (
                (50 + self.airtime())
                * (consts.pow_cons[1] + consts.pow_cons[2])
                * consts.V
                / 1e6
            )
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
            consts.Lpld0
            + 10 * consts.gamma * math.log10(self.dist(destination) / consts.d0)
            + np.random.normal(-consts.variance, consts.variance)
        )
        Prx = consts.Ptx - consts.GL - Lpl
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
            consts.nr_lost += 1
            if not self.node.is_gateway():
                self.node.nr_lost += 1

        if self.collided:
            consts.nr_collisions += 1

        if self.is_received():
            consts.nr_received += 1

        if self.processed:
            consts.nr_processed += 1

        if self.sent:
            consts.nr_packets_sent += 1

        consts.erx += self.energy_receive()
        consts.etx += self.energy_transmit()
        consts.total_energy += self.energy_transmit() + self.energy_receive()

    def is_received(self):
        return not self.collided and self.processed and not self.lost

    def was_sent_to(self, node):
        return self.receiver is node

    def check_collision(self, env):
        self.processed = True
        if BroadcastTraffic.nr_data_packets > consts.max_packets:
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
                        self.node.nr_collisions += 1
                        self.processed = False

                    if other.processed and other.was_sent_to(self.node):
                        log(
                            env,
                            f"[PACKET-DROP-OTHER] {other} from {other.node} is dropped",
                        )
                        other.node.nr_collisions += 1
                        other.processed = False

                if (
                    frequency_collision(self, other)
                    and sf_collision(self, other)
                    and timing_collision(self, other, env)
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
        super().__init__(node, consts.data_gateway)
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
            consts.nr_data_packets_sent += 1

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
            consts.nr_sack_sent += 1

    def was_sent_to(self, node):
        return self.sf == node.sf

    def __str__(self):
        return "SACK packet"
