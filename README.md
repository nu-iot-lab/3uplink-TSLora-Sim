# 3uplink TSLora Simulator

## Description


The 3uplink TSLora Simulator project, developed for the NU IoT Lab, focuses on simulating the transmission and reception of data over LoRa (Long Range) networks. This project aims to provide a robust platform for testing and developing IoT applications that rely on LoRa technology for communication. Simulator introduces an advanced Reinforcement Learning (RL) functionality, based on the Deep Q Network (DQN) algorithm, to automatically choose the best number of uplinks (from 1 to 3) for nodes to transmit data, optimizing for the best Packet Reception Rate (PRR) value.


## Installation

To get started with the 3uplink TSLora Simulator, follow these steps:

1. Clone the repository:

```
git clone https://github.com/nu-iot-lab/3uplink-TSLora-Sim.git
```

2. Navigate into the project directory:

```
cd 3uplink-TSLora-Sim
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```


4. Install our custom Gymnasium environment for LoRa:

```
pip install -e loraenv
```

## Usage

To run the simulation, execute the following command from the main directory:

```
python main.py --nodes_count=<number_of_nodes> --data_size=<size_of_data_in_bytes> --avg_wake_up_time=<average_wake_up_time_in_seconds> --sim_time=<simulation_time_in_seconds>
```

Default parameter values:

-   `--nodes_count=10`: The number of nodes participating in the simulation.
-   `--data_size=16`: The size of the data to be transmitted, in bytes.
-   `--avg_wake_up_time=30`: The average delay, in seconds, before nodes wake up from the simulation start to begin transmitting data.
-   `--sim_time=3600`: The total simulation time, in seconds.

## Features

-   Simulates LoRa network behaviour for uplink data transmission.
-   Supports various simulation parameters to mimic real-world IoT environments.

-   Integrates a Reinforcement Learning (RL) mechanism, utilizing the Deep Q Network (DQN) algorithm, to dynamically optimize the number of uplinks for nodes, aiming to achieve the best Packet Reception Rate (PRR).

-   Includes visualization tools for analyzing the simulation results.

## License

This project is licensed under the GNU General Public License - see the LICENSE.md file for details.

## Credits

The 3uplink TSLora Simulator project is developed and maintained by the NU IoT Lab.