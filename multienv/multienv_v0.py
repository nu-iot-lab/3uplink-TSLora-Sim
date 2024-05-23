from multienv.env.multienv import LoRaEnvParallel
from pettingzoo.utils import parallel_to_aec

def env(**kwargs): 
    env_cur = LoRaEnvParallel(**kwargs)
    env_cur = parallel_to_aec(env_cur)
    return env_cur