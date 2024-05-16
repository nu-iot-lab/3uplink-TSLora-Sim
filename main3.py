import os
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
import logging
import sys

from multienv.multienv_v0 import env

logging.basicConfig(level=logging.INFO)

# Function to create the environment
def create_env(config):
    logging.info(f"Creating environment with config: {config}")
    env_instance = env(**config)
    logging.info(f"Custom Env possible_agents: {env_instance.possible_agents}")
    return PettingZooEnv(env_instance)

# Function to train the environment
def train_fn(config):
    logging.info("Registering environment.")
    register_env("LoRaEnvParallel", create_env)

    # Create a test environment to get observation and action spaces
    test_env = create_env(config)
    logging.info(f"Wrapped Env possible_agents: {test_env.env.possible_agents}")

    # Check if possible_agents exists
    if hasattr(test_env.env, 'possible_agents'):
        obs_space = test_env.env.observation_space(test_env.env.possible_agents[0])
        act_space = test_env.env.action_space(test_env.env.possible_agents[0])
    else:
        raise AttributeError("The environment does not have 'possible_agents' attribute.")

    algo_config = (
        DQNConfig()
        .environment(env="LoRaEnvParallel", env_config=config)
        .env_runners(num_env_runners=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
            hiddens=[],
            dueling=False,
        )
        .multi_agent(
            policies={agent: (None, obs_space, act_space, {}) for agent in test_env.env.possible_agents},
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .framework(framework="torch")
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 0.1,
                "final_epsilon": 0.0,
                "epsilon_timesteps": 100000,
            }
        )
    )

    try:
        tune.run(
            "DQN",
            name="DQN_LoRaEnvParallel",
            stop={"timesteps_total": 1000000},
            checkpoint_freq=10,
            config=algo_config.to_dict(),
        )
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 main3.py <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>")
        sys.exit(1)
    
    nodes_count = int(sys.argv[1])
    data_size = int(sys.argv[2])
    avg_wake_up_time = int(sys.argv[3])
    sim_time = int(sys.argv[4])

    ray.init()

    try:
        analysis = tune.run(
            train_fn,
            config={
                "nodes_count": nodes_count,
                "data_size": data_size,
                "avg_wake_up_time": avg_wake_up_time,
                "sim_time": sim_time
            },
            metric="episode_reward_mean",
            mode="max"
        )

        print("Best checkpoint:", analysis.best_checkpoint)

        with analysis.best_checkpoint.as_directory() as tmpdir:
            trainer = DQNConfig.load_from_checkpoint(tmpdir)
    except Exception as e:
        logging.error(f"An error occurred during the Ray Tune run: {e}")
