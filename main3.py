import os
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

from multienv.multienv_v0 import env

if __name__ == "__main__":
    ray.init()

    # Register the environment
    register_env("LoRaEnvParallel", lambda config: PettingZooEnv(env(config)))

    # Create a test environment to get observation and action spaces
    test_env = PettingZooEnv(env({
        "nodes_count": 10,
        "data_size": 16,
        "avg_wake_up_time": 30,
        "sim_time": 3600
    }))
    obs_space = test_env.observation_space(test_env.possible_agents[0])
    act_space = test_env.action_space(test_env.possible_agents[0])

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
            policies={agent: (None, obs_space, act_space, {}) for agent in test_env.possible_agents},
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

    tune.run(
        "DQN",
        name="DQN_LoRaEnvParallel",
        stop={"timesteps_total": 1000000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )
