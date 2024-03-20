import simulator.consts as consts

from random import gauss, uniform
from simulator.communications import DataPacket, SackPacket

class Frame:
    def __init__(self, sf):
        self.sf = sf
        self.data_p_rec_time = DataPacket(sf).rec_time
        self.min_frame_length = (
            3 * 100 * self.data_p_rec_time
        )  # 3 slots for each packet
        # self.guard_time = 3 * 0.0001 * self.min_frame_length
        self.guard_time = 2  # 2ms
        self.min_nr_slots = int(
            self.min_frame_length / (self.data_p_rec_time + 2 * self.guard_time)
        )

        self.nr_slots = self.min_nr_slots
        self.nr_taken_slots = 0
        self.nr_slots_SACK = self.nr_slots

        self.frame_length = self.min_frame_length

        self.sack_p_rec_time = SackPacket(self.nr_slots, sf).rec_time
        self.data_slot_len = self.data_p_rec_time + 2 * self.guard_time
        self.sack_slot_len = self.sack_p_rec_time + 2 * self.guard_time

        self.trans_time = uniform(0, self.frame_length - self.sack_slot_len)
        self.trans_time_period = self.sack_p_rec_time + self.guard_time
        self.next_round_start_time = self.trans_time + self.trans_time_period + 1

        self.slots = [None for _ in range(self.nr_slots)]

    def assign_slots(self):
        slots = [None for _ in range(self.nr_slots)]
        for node in consts.nodes:
            self.add(node)
        return slots

    def check_data_collision(self, env):
        drifting_times = {}
        for i in range(1, self.nr_taken_slots):
            if self.slots[i] is not None and self.slots[i - 1] is not None:
                # generate drifting time for the current slot if it hasn't been generated before
                if i not in drifting_times:
                    drifting_times[i] = gauss(0, 1.0)

                df = drifting_times[i]
                start_time_n = env.now + self.data_slot_len * i + df + self.guard_time
                end_time_n = (
                    env.now + self.data_slot_len * (i + 1) - self.guard_time + df
                )

                # generate drifting time for the previous slot if it hasn't been generated before
                if i - 1 not in drifting_times:
                    drifting_times[i - 1] = gauss(0, 1.0)

                df_prev = drifting_times[i - 1]
                start_time_prev = (
                    env.now + self.data_slot_len * (i - 1) + df_prev + self.guard_time
                )
                end_time_prev = (
                    env.now + self.data_slot_len * i - self.guard_time + df_prev
                )

                if start_time_n <= end_time_prev and start_time_prev <= end_time_n:
                    consts.nr_data_collisions += 1

    def __update_fields(self):
        # self.sack_p_rec_time = SackPacket(self.nr_slots, self.sf)
        sack_packet = SackPacket(self.nr_slots, self.sf)
        self.sack_p_rec_time = sack_packet.airtime()
        if self.nr_taken_slots > self.min_nr_slots:
            self.frame_length = (
                self.nr_slots * self.data_p_rec_time + self.sack_p_rec_time
            ) / (1.0 - 6 * 0.0001 * (self.nr_slots + 1))
        # self.guard_time = 3 * 0.0001 * self.frame_length
        self.guard_time = 2
        self.data_slot_len = self.data_p_rec_time + 2 * self.guard_time
        self.sack_slot_len = self.sack_p_rec_time + 2 * self.guard_time

    def next_frame(self):
        self.trans_time = (
            self.next_round_start_time
            + self.frame_length
            - self.sack_slot_len
            + self.guard_time
        )
        self.trans_time_period = self.sack_p_rec_time + self.guard_time
        self.next_round_start_time = self.trans_time + self.trans_time_period + 1
        self.nr_slots_SACK = self.nr_slots

    def add(self, node):
        if self.nr_taken_slots < self.nr_slots:
            slot_index = self.slots.index(None)
            node.slot = [slot_index, slot_index + 1, slot_index + 2]
            self.slots[slot_index] = node
            self.slots[slot_index + 1] = node
            self.slots[slot_index + 2] = node
        else:
            node.slot = [self.nr_slots, self.nr_slots + 1, self.nr_slots + 2]
            self.slots.append(node)
            self.slots.append(node)
            self.slots.append(node)
            self.nr_slots += 3
            self.__update_fields()
        self.nr_taken_slots += 3

    def remove(self, node):
        if node.slot[0] is None and node.slot[1] is None and node.slot[2] is None:
            return
        self.slots[node.slot[0]] = None
        self.slots[node.slot[1]] = None
        self.slots[node.slot[2]] = None
        node.slot[0] = None
        node.slot[1] = None
        node.slot[2] = None

    def __str__(self):
        slots = [str(node) if node is not None else "None" for node in self.slots]
        return f"Frame: {slots}"
