import datetime
import time

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow_probability.python.distributions.categorical import Categorical
from abc import ABC, abstractmethod
import random
import numpy as np
import os
import gym


class MaslouRLModelPPODiscrete(ABC):
    def __init__(self):
        self.env = self.make_env(0, 0, False, 0, "temp")()
        self.input_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.model = self.build_model()

    def summary(self):
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)
        self.model.summary()


    @abstractmethod
    def make_env(self, seed, idx, capture_video, capture_every_n_video, run_name):
        pass

    @abstractmethod
    def build_model(self) -> keras.Model:
        pass

    def get_action_and_value(self, x, action=None):
        model_res = self.model(x)
        probs = model_res[0]
        distribution = Categorical(probs=probs)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), model_res[1]

    def get_value(self, x):
        model_res = self.model(x)
        return model_res[1]

    def train(self, learning_rate=2.5e-4, num_steps=128,
              num_envs=4, seed=42, capture_video=True,
              capture_every_n_video=50, run_name="PPO_run_name",
              total_timesteps=1000000, anneal_lr=True, gae=True,
              discount_gamma=0.99, gae_lambda=0.95, update_epochs=4,
              minibatches=4, norm_adv=True, clip_coef=0.2, clip_vloss=True,
              ent_coef=0.01, vf_coef=0.5, track=False, wandb_project_name=None, wandb_entity=None,max_grad_norm=0.5, config=None):
        batch_size = num_steps * num_envs
        envs = gym.vector.SyncVectorEnv([self.make_env(seed + i,
                                                       i,
                                                       capture_video,
                                                       capture_every_n_video,
                                                       run_name) for i in range(num_envs)])
        if track:
            import wandb

            wandb.init(project=wandb_project_name,
                       entity=wandb_entity,
                       sync_tensorboard=True,
                       config=vars(config),
                       name=run_name,
                       monitor_gym=True,
                       save_code=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + run_name
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["TF_CUDNN_DETERMINISTIC"] = '1'

        optimizer = Adam(learning_rate=learning_rate, epsilon=1e-5)
        obs = np.zeros((num_steps, num_envs) + envs.single_observation_space.shape, dtype=float)
        actions = np.zeros((num_steps, num_envs) + envs.single_action_space.shape, dtype=float)
        logprobs = np.zeros((num_steps, num_envs), dtype=float)
        rewards = np.zeros((num_steps, num_envs), dtype=float)
        dones = np.zeros((num_steps, num_envs), dtype=float)
        values = np.zeros((num_steps, num_envs), dtype=float)

        global_step = 0
        start_time = time.time()
        next_obs = envs.reset()
        next_done = np.zeros(num_envs)
        num_updates = total_timesteps // batch_size
        for update in range(1, num_updates + 1):
            # print(f"Start update, {update}")
            summary_writer.flush()
            if anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                lrnow = frac * learning_rate
                optimizer._lr = lrnow
            for step in range(0, num_steps):
                global_step += num_envs
                obs[step] = next_obs
                dones[step] = next_done

                action, logprob, entropy, value = self.get_action_and_value(next_obs)
                values[step] = tf.squeeze(value)
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, done, info = envs.step(action.numpy())
                # print(reward)
                # exit(0)
                next_done = done
                rewards[step] = tf.squeeze(reward)
                if "episode" in info.keys():
                    for item in info["episode"]:
                        if item is not None:
                            with summary_writer.as_default():
                                tf.summary.scalar("charts/episodic_return", item["r"], global_step)
                                tf.summary.scalar("charts/episodic_length", item["l"], global_step)
                            print(f"global_step={global_step}, episodic_return={item['r']}")
                            break

            next_value = tf.reshape(self.get_value(next_obs), (1, -1))

            if gae:
                advantages = np.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + discount_gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + discount_gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            else:
                returns = np.zeros_like(rewards)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = rewards[t] + discount_gamma * nextnonterminal * next_return
                advantages = returns - values

            b_obs = tf.convert_to_tensor(obs.reshape((-1,) + envs.single_observation_space.shape), dtype=tf.float32)
            b_logprobs = tf.convert_to_tensor(logprobs.reshape(-1), dtype=tf.float32)
            b_actions = tf.convert_to_tensor(actions.reshape((-1,) + envs.single_action_space.shape), dtype=tf.int32)
            b_advanatges = tf.convert_to_tensor(advantages.reshape(-1), dtype=tf.float32)
            b_returns = tf.convert_to_tensor(returns.reshape(-1), dtype=tf.float32)
            b_values = tf.convert_to_tensor(values.reshape(-1),  dtype=tf.float32)

            b_inds = np.arange(batch_size)
            minibatch_size = batch_size // minibatches

            self.model.build(input_shape = (4,))
            for epoch in range(update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = tf.convert_to_tensor(b_inds[start:end], dtype=tf.int32)
                    mb_obs = tf.gather(b_obs, indices=mb_inds)
                    mb_actions = tf.gather(b_actions, indices=mb_inds)
                    mb_logprobs = tf.gather(b_logprobs, indices=mb_inds)
                    mb_advantages = tf.gather(b_advanatges, indices=mb_inds)
                    mb_returns = tf.gather(b_returns, indices=mb_inds)
                    mb_values = tf.gather(b_values, indices=mb_inds)
                    with tf.GradientTape() as tape:
                        _, newlogprob, entropy, newvalue = self.get_action_and_value(mb_obs, mb_actions)

                        log_ratio = newlogprob - mb_logprobs

                        ratio = tf.math.exp(log_ratio)


                        if norm_adv:
                            mb_advantages = (mb_advantages - tf.math.reduce_mean(mb_advantages)) / (tf.math.reduce_std(mb_advantages) + 1e-8)

                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * tf.clip_by_value(ratio, 1 - clip_coef, 1 + clip_coef)
                        pg_loss = tf.math.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

                        newvalue = tf.squeeze(newvalue)
                        if clip_vloss:
                            v_loss_unclipped = (newvalue - mb_returns) ** 2
                            v_clipped = mb_values + tf.clip_by_value(newvalue - mb_returns, -clip_coef, clip_coef)
                            v_loss_clipped = (v_clipped - mb_returns) ** 2
                            v_loss_max = tf.maximum(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * tf.math.reduce_mean(v_loss_max)

                        else:
                            v_loss = 0.5 * tf.math.reduce_mean((newvalue - mb_returns)**2)

                        entropy_loss = tf.math.reduce_mean(entropy)

                        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                    feature_network_gradient = tape.gradient(loss, self.model.trainable_variables)
                    feature_network_gradient, _ = tf.clip_by_global_norm(feature_network_gradient, max_grad_norm)
                    optimizer.apply_gradients(zip(feature_network_gradient, self.model.trainable_variables))

            y_pred, y_true = b_values, b_returns
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            with summary_writer.as_default():
                tf.summary.scalar("charts/learning_rate", optimizer._lr, global_step)
                tf.summary.scalar("losses/value_loss", v_loss, global_step)
                tf.summary.scalar("losses/policy_loss", pg_loss, global_step)
                tf.summary.scalar("losses/entropy_loss", entropy_loss, global_step)
                tf.summary.scalar("losses/explained_variance", explained_var, global_step)
                tf.summary.scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
