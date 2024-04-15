from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        episode_rewards = self.model.ep_info_buffer
        if episode_rewards:
            # Collect the total reward for the last episode
            total_reward = episode_rewards[-1]['r']
            self.rewards.append(total_reward)
            print(f"Episode: {len(self.rewards)} Reward: {total_reward}")
