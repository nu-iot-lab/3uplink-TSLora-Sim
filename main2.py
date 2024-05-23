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
from ray.tune.logger import TBXLoggerCallback
from torch.utils.tensorboard import SummaryWriter

from multienv.multienv_v0 import env

logging.basicConfig(level=logging.INFO)

class CustomMetricsCallback(tune.Callback):
    def __init__(self, logdir):
        self.logdir = logdir
        self.writer = SummaryWriter(log_dir=logdir)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_unwrapped()[0]
        total_steps = episode.length
        total_uplinks = sum(episode.custom_metrics[agent]["uplink_attempts"] for agent in env.possible_agents)
        total_reward = sum(episode.reward[agent] for agent in env.possible_agents)
        uplinks_per_node = {agent: episode.custom_metrics[agent]["uplink_attempts"] for agent in env.possible_agents}

        uplinks_per_step = total_uplinks / total_steps if total_steps > 0 else 0
        reward_per_uplink = total_reward / total_uplinks if total_uplinks > 0 else 0

        episode.custom_metrics["total_uplinks"] = total_uplinks
        episode.custom_metrics["total_steps"] = total_steps
        episode.custom_metrics["uplinks_per_step"] = uplinks_per_step
        episode.custom_metrics["reward_per_uplink"] = reward_per_uplink
        episode.custom_metrics["uplinks_per_node"] = uplinks_per_node

        logging.info(f"Episode {episode.episode_id} ended with {total_uplinks} total uplinks, "
                     f"{total_steps} steps, {uplinks_per_step:.4f} uplinks per step, "
                     f"{reward_per_uplink:.4f} reward per uplink, and {uplinks_per_node} uplink attempts.")

        # Log custom metrics to TensorBoard
        self.writer.add_scalar("Metrics/Total_Uplinks", total_uplinks, episode.episode_id)
        self.writer.add_scalar("Metrics/Reward_Per_Uplink", reward_per_uplink, episode.episode_id)
        
        for agent, uplinks in uplinks_per_node.items():
            self.writer.add_scalar(f"Metrics/Uplinks_Per_Node/{agent}", uplinks, episode.episode_id)

    def on_trial_end(self, iteration, trials, trial, **info):
        self.writer.flush()

    def on_experiment_end(self, **kwargs):
        self.writer.close()

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
            .resources(num_gpus=0)
            .framework(framework="torch")
        )

        try:
            logdir = "~/ray_results"
            tune.run(
                DQN,
                name="DQN_LoRaEnvParallel",
                stop={"timesteps_total": 100000},
                checkpoint_freq=10,
                config=config.to_dict(),
                local_dir=logdir,  # Specify the directory for logging
                callbacks=[TBXLoggerCallback(), CustomMetricsCallback(logdir)],
                log_to_file=True,
            )
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise
    else:
        print(
            "usage: python main3.py <number_of_nodes> <data_size(bytes)> <avg_wake_up_time(secs)> <sim_time(secs)>"
        )
        exit(-1)
