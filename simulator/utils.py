import simulator.consts as consts

from simulator.singleton import ArgumentSingleton, DataGatewaySingleton

args = ArgumentSingleton.get_instance()
nodes_count = args.nodes_count


def frequency_collision(p1, p2):
    if abs(p1.freq - p2.freq) <= 120 and (p1.bw == 500 or p2.freq == 500):
        return True
    elif abs(p1.freq - p2.freq) <= 60 and (p1.bw == 250 or p2.freq == 250):
        return True
    else:
        if abs(p1.freq - p2.freq) <= 30:
            return True
    return False


def sf_collision(p1, p2):
    return p1.sf == p2.sf


def power_collision(p1, p2):
    p1_rssi, p2_rssi = p1.rssi(p2.node), p2.rssi(p1.node)
    if abs(p1_rssi - p2_rssi) < consts.power_threshold:
        # packets are too close to each other, both collide
        # return both packets as casualties
        return p1, p2
    elif p1_rssi - p2_rssi < consts.power_threshold:
        # p2 overpowered p1, return p1 as casualty
        return (p1,)
    # p2 was the weaker packet, return it as a casualty
    return (p2,)


def timing_collision(p1, p2, env):
    # assuming p1 is the freshly arrived packet and this is the last check
    # we've already determined that p1 is a weak packet, so the only
    # way we can win is by being late enough (only the first n - 5 preamble symbols overlap)

    # assuming 8 preamble symbols

    # we can lose at most (Npream - 5) * Tsym of our preamble
    Tpreamb = 2**p1.sf / (1.0 * p1.bw) * (consts.npream - 5)

    # check whether p2 ends in p1's critical section
    p2_end = p2.add_time + p2.rec_time
    p1_cs = env.now + Tpreamb
    if p1_cs < p2_end:
        # p1 collided with p2 and lost
        return True
    return False


def get_sensitivity(sf, bw):
    return consts.sensitivities[sf - 7, [125, 250, 500].index(bw) + 1]


def log(env, str):
    print(f'{f"{env.now / 1000:.3f} s":<12} {str}')


def reset_simulator():
    data_gateway = DataGatewaySingleton.reset()
    consts.nodes = []
    consts.nr_collisions = 0
    consts.nr_data_collisions = 0
    consts.nr_received = 0
    consts.nr_received_data_packets = 0
    consts.nr_processed = 0
    consts.nr_lost = 0
    consts.nr_packets_sent = 0
    consts.nr_data_packets_sent = 0
    consts.nr_retransmission = 0
    consts.nr_data_retransmissions = 0
    consts.nr_sack_sent = 0
    consts.nr_sack_missed_count = 0
    consts.total_energy = 0
    consts.erx = 0
    consts.etx = 0


def show_final_statistics():
    consts.nr_data_retransmissions = (
        consts.nr_sack_missed_count + consts.nr_lost + consts.nr_data_collisions
    )

    consts.nr_received_data_packets = 0

    print("\n!-- NODE STATISTICS --!\n")
    sum = 0
    for node in consts.nodes:
        consts.nr_received_data_packets += node.packets_received_count
        node.calculate_prr()
        sum += node.calculate_prr()
        print(
            f"NODE {node.node_id}: "
            f"PRR - {node.calculate_prr():.3f}, "
            f"{node.packets_sent_count:3} packets sent, "
            f"{node.packets_received_count:3} packets received, "
            f"{node.missed_sack_count:3} SACKs missed, "
            f"{node.nr_lost:3} packets lost, "
            f"{node.nr_collisions:3} collisions"
        )
    print(f"Average PRR: {(sum / nodes_count):.3f}")

    print("\n!-- TOTAL STATISTICS --!\n")
    print("Data collisions:", consts.nr_data_collisions)
    print("Lost packets (due to path loss):", consts.nr_lost)
    print("Transmitted data packets:", consts.nr_data_packets_sent)
    print("Received data packets:", consts.nr_received_data_packets)
    print("Transmitted SACK packets:", consts.nr_sack_sent)
    print("Missed SACK packets:", consts.nr_sack_missed_count)
    print("Data Retransmissions:", consts.nr_data_retransmissions)
    print(f"Average energy consumption (Rx): {(consts.erx / nodes_count):.3f} J")
    print(f"Average energy consumption (Tx): {(consts.etx / nodes_count):.3f} J")
    print(
        f"Average energy consumption per node: {consts.total_energy / nodes_count:.3f} J"
    )
    print(
        f"Network PRR (version 1): {(consts.nr_data_packets_sent - consts.nr_data_retransmissions) / consts.nr_data_packets_sent:.3f}"
    )
    print(
        f"Network PRR (version 2): {(consts.nr_received_data_packets / consts.nr_data_packets_sent):.3f}"
    )
