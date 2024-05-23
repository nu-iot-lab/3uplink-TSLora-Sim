import os
import sys
import ray
import logging
import simulator.consts as consts
import matplotlib.pyplot as plt

from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.tune.logger import TBXLoggerCallback

from multienv.multienv_v0 import env

logging.basicConfig(level=logging.INFO)

def plot_metrics(df):
    # Create a figure with 2 subplots arranged vertically
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot mean episode reward on the first subplot
    if 'episode_reward_mean' in df.columns:
        df['episode_reward_mean'].plot(ax=ax[0])
        ax[0].set_title('Mean Episode Reward')
        ax[0].set_xlabel('Training Iterations')
        ax[0].set_ylabel('Reward')
    else:
        logging.warning("No 'episode_reward_mean' column found in results.")
        ax[0].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

    # Plot mean episode length on the second subplot
    if 'episode_len_mean' in df.columns:
        df['episode_len_mean'].plot(ax=ax[1])
        ax[1].set_title('Mean Episode Length')
        ax[1].set_xlabel('Training Iterations')
        ax[1].set_ylabel('Length')
    else:
        logging.warning("No 'episode_len_mean' column found in results.")
        ax[1].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

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
                    "epsilon_timesteps": 1000000,
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
            .resources(num_gpus=0)
            .framework(framework="torch")
        )

        try:
            analysis = tune.run(
                DQN,
                name="DQN_LoRaEnvParallel",
                stop={"timesteps_total": 1000},
                checkpoint_freq=100,
                keep_checkpoints_num=5,
                checkpoint_score_attr="training_iteration",
                config=config.to_dict(),
                local_dir="~/ray_results",  # Specify the directory for logging
                callbacks=[TBXLoggerCallback()],
                log_to_file=True,
            )

            # Get the best trial
            best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")

            # Use TensorBoard to visualize results
            print(f"Training completed. Use TensorBoard to visualize results: tensorboard --logdir {best_trial.local_path}")

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise
    else:
        print(
            "usage: python main3.py <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>"
        )
        exit(-1)
