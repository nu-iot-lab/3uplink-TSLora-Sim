from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.total_rewards = 0

    def _on_step(self) -> bool:
        # Add the rewards from the last step to total_rewards
        self.total_rewards += self.locals["rewards"]

        # Check for the 'done' flag in the info dict which indicates the episode is done
        info = self.locals["infos"]
        # Assume single environment for simplicity; adjust if using VecEnv
        if "is_terminal" in info[0] or info[0].get("terminal_observation") is not None:
            self.episode_rewards.append(self.total_rewards)
            self.total_rewards = 0  # Reset the reward counter after the episode ends

        return True  # Return True to continue training, False will stop the training

    def _on_training_end(self):
        # Optionally implement this to perform actions after training ends
        pass
