import numpy as np
from collections import defaultdict

# Arrays of measured sensitivfity values
sf7 = np.array([7, -123.0, -120.0, -117.0])
sf8 = np.array([8, -126.0, -123.0, -120.0])
sf9 = np.array([9, -129.0, -126.0, -123.0])
sf10 = np.array([10, -132.0, -129.0, -126.0])
sf11 = np.array([11, -134.53, -131.52, -128.51])
sf12 = np.array([12, -137.0, -134.0, -131.0])
sensitivities = np.array([sf7, sf8, sf9, sf10, sf11, sf12])

# IsoThresholds for collision detection caused by imperfect orthogonality of SFs
IS7 = np.array([1, -8, -9, -9, -9, -9])
IS8 = np.array([-11, 1, -11, -12, -13, -13])
IS9 = np.array([-15, -13, 1, -13, -14, -15])
IS10 = np.array([-19, -18, -17, 1, -17, -18])
IS11 = np.array([-22, -22, -21, -20, 1, -20])
IS12 = np.array([-25, -25, -25, -24, -23, 1])
iso_thresholds = np.array([IS7, IS8, IS9, IS10, IS11, IS12])

# power consumptions for transmitting, receiving, and operating in mA
pow_cons = [75, 45, 30]
V = 3.3  # voltage XXX

# global
data_gateway = None
nodes = []

coding_rate = 1
drifting_range = [-0.2, 0.2]
mean = 0  # Mean of the normal distribution
std_dev = 0.0001  # Standard deviation of the normal distribution

# Statistics
nr_collisions = 0
nr_data_collisions = 0
nr_received = 0
nr_received_data_packets = 0
nr_processed = 0
nr_lost = 0
nr_packets_sent = 0
nr_data_packets_sent = 0
nr_retransmission = 0
nr_data_retransmissions = 0
nr_sack_sent = 0
nr_sack_missed_count = 0

total_energy = 0

erx = 0
etx = 0

Ptx = 14
gamma = 2.08
d0 = 40.0
var = 0
Lpld0 = 127.41
GL = 0
power_threshold = 6
npream = 8
max_packets = 500
retrans_count = 10000
sigma = 0.38795
variance = 1.0

# min waiting time before retransmission in s
min_wait_time = 4.5

# max distance between nodes and base station
max_dist = 180

# base station position
bsx = max_dist + 10
bsy = max_dist + 10
x_max = bsx + max_dist + 10
y_max = bsy + max_dist + 10

