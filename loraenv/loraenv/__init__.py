from gymnasium.envs.registration import register

register(
    id='loraenv/LoRa-v0',
    entry_point='loraenv.envs:LoRaEnv', 
)