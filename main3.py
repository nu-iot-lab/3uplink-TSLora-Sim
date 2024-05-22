import os
import sys
import ray
import logging
import simulator.consts as consts

from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

from multienv.multienv_v0 import env

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        nodes_count = int(sys.argv[1])
        data_size = int(sys.argv[2])
        avg_wake_up_time = int(sys.argv[3])
        sim_time = int(sys.argv[4])

        consts.nodes_count = nodes_count
        consts.data_size = data_size
        consts.avg_wake_up_time = avg_wake_up_time
        consts.sim_time = sim_time

        ray.init()

        # Register the environment
        def create_env(config):
            env_instance = env(
                nodes_count=nodes_count,
                data_size=data_size,
                avg_wake_up_time=avg_wake_up_time,
                sim_time=sim_time,
            )
            logging.info(f"Custom Env possible_agents: {env_instance.possible_agents}")
            return PettingZooEnv(env_instance)

        register_env("LoRaEnvParallel", create_env)

        # Create a test environment to get observation and action spaces
        test_env = create_env(
            {
                "nodes_count": nodes_count,
                "data_size": data_size,
                "avg_wake_up_time": avg_wake_up_time,
                "sim_time": sim_time,
            }
        )
        logging.info(f"Wrapped Env possible_agents: {test_env.env.possible_agents}")

        # Check if possible_agents exists
        if hasattr(test_env.env, "possible_agents"):
            obs_space = test_env.env.observation_space(test_env.env.possible_agents[0])
            act_space = test_env.env.action_space(test_env.env.possible_agents[0])
        else:
            raise AttributeError(
                "The environment does not have 'possible_agents' attribute."
            )

        config = (
            DQNConfig()
            .environment(
                env="LoRaEnvParallel",
                env_config={
                    "nodes_count": nodes_count,
                    "data_size": data_size,
                    "avg_wake_up_time": avg_wake_up_time,
                    "sim_time": sim_time,
                },
            )
            .env_runners(
                num_env_runners=1,
                rollout_fragment_length=30,
                exploration_config={
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 0.1,
                    "final_epsilon": 0.0,
                    "epsilon_timesteps": 100000,
                },
            )
            .training(
                train_batch_size=200,
                hiddens=[],
                dueling=False,
            )
            .multi_agent(
                policies={
                    agent: (None, obs_space, act_space, {})
                    for agent in test_env.env.possible_agents
                },
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            )
            .resources(num_gpus=1)
            .framework(framework="torch")
        )

        try:
            tune.run(
                DQN,
                name="DQN_LoRaEnvParallel",
                stop={"timesteps_total": 1000000},
                checkpoint_freq=10,
                config=config.to_dict(),
            )
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise
    else:
        print(
            "usage: python main3.py <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>"
        )
        exit(-1)
