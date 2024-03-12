from gymnasium.envs.registration import register

register(
    id='lora-v0',
    entry_point='envs.environment:LoRaEnv',
    kwargs={'num_agents': 10, 'data_size': 100, 'avg_wake_up_time': 5, 'sim_time': 100}
)