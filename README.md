# 3uplink TSLora Simulator

## Description

The 3uplink TSLora Simulator project, developed for the NU IoT Lab, focuses on simulating the transmission and reception of data over LoRa (Long Range) networks. This project aims to provide a robust platform for testing and developing IoT applications that rely on LoRa technology for communication.

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

## Usage

To run the simulation, execute the following command from the 'simulator' directory:

```
python main.py --nodes_count=10 --data_size=16 --avg_wake_up_time=30 --sim_time=3600
```

## Features

-   Simulates LoRa network behaviour for uplink data transmission.
-   Supports various simulation parameters to mimic real-world IoT environments.
-   Includes visualization tools for analyzing the simulation results.

## License

This project is licensed under the GNU General Public License - see the LICENSE.md file for details.

## Credits

The 3uplink TSLora Simulator project is developed and maintained by the NU IoT Lab.
