import os
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
import logging

from multienv.multienv_v0 import env

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    ray.init()

    # Register the environment
    def create_env(config):
        env_instance = env()
        logging.info(f"Custom Env possible_agents: {env_instance.possible_agents}")
        return PettingZooEnv(env_instance)

    register_env("LoRaEnvParallel", create_env)

    # Create a test environment to get observation and action spaces
    test_env = create_env({})
    logging.info(f"Wrapped Env possible_agents: {test_env.env.possible_agents}")

    # Check if possible_agents exists
    if hasattr(test_env.env, 'possible_agents'):
        obs_space = test_env.env.observation_space(test_env.env.possible_agents[0])
        act_space = test_env.env.action_space(test_env.env.possible_agents[0])
    else:
        raise AttributeError("The environment does not have 'possible_agents' attribute.")

    config = (
        DQNConfig()
        .environment(env="LoRaEnvParallel", env_config={
            "nodes_count": 10,
            "data_size": 16,
            "avg_wake_up_time": 30,
            "sim_time": 3600
        })
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
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
            config=config.to_dict(),
        )
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise
