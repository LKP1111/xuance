import torch
from triton.language import dtype

from xuance.common import Tuple, Union
from xuance.torch.learners import Learner
from xuance.torch.policies import DreamerV3Policy
from xuance.torch.utils import kl_div, dotdict
from argparse import Namespace
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from torch.cuda.amp import autocast, GradScaler


class DreamerV3_Learner(Learner):
    def __init__(self,
                 config: dotdict,
                 policy: DreamerV3Policy,
                 action_shape: Union[int, Tuple[int, ...]]):
        super(DreamerV3_Learner, self).__init__(config, policy)
        self.policy = t = policy  # for code completion
        self.action_shape = action_shape

        # config
        self.is_continuous = config.is_continuous
        self.tau = config.soft_update_rate
        self.gamma = config.gamma
        self.soft_update_freq = config.soft_update_freq

        self.kl_dynamic = config.loss_scales.kl_dynamic  # 1.0
        self.kl_representation = config.loss_scales.kl_representation  # 0.1
        self.kl_free_nats = config.loss_scales.kl_free_nats  # 1.0
        self.kl_regularizer = config.loss_scales.kl_regularizer  # 1.0
        self.continue_scale_factor = config.loss_scales.continue_scale_factor  # 1.0

        wm_list = [t.encoder, t.rssm, t.decoder, t.reward_predictor, t.discount_predictor]
        self.wm_params = sum([list(x.parameters()) for x in wm_list], [])
        if config.harmony:
            self.wm_params += [
                policy.harmonizer_s1.get_harmony(),
                policy.harmonizer_s2.get_harmony(),
                policy.harmonizer_s3.get_harmony()
            ]
        # optimizers
        self.optimizer = {
            'model': torch.optim.Adam(self.wm_params, config.learning_rate_model),
            'actor': torch.optim.Adam(policy.actor.parameters(), config.learning_rate_actor),
            'critic': torch.optim.Adam(policy.critic.parameters(), config.learning_rate_critic)
        }
        # AMP GradScaler for float16
        self.scaler = GradScaler()
        self.gradient_step = 0

    def update(self, **samples):
        if self.gradient_step % self.soft_update_freq == 0:
            self.policy.soft_update(self.tau)
        obs = torch.as_tensor(samples['obs'], device=self.device, dtype=torch.float32)
        acts = torch.as_tensor(samples['acts'], device=self.device)
        rews = torch.as_tensor(samples['rews'], device=self.device)
        terms = torch.as_tensor(samples['terms'], device=self.device)
        truncs = torch.as_tensor(samples['truncs'], device=self.device)  # no use
        is_first = torch.as_tensor(samples['is_first'], device=self.device)
        """
        seq_shift (preprocess)
        (o1, a1 -> a0, r1, d1, f1)
        """
        if not self.is_continuous:
            acts = torch.nn.functional.one_hot(acts.long(), num_classes=self.action_shape).float()
        is_first[0, :] = torch.ones_like(is_first[0, :])
        acts = torch.cat((torch.zeros_like(acts[:1]), acts[:-1]), 0)  # bug fixed ones_like -> zeros_like
        cont = 1 - terms

        with autocast(dtype=torch.float16):
            po, pr, pc, priors_logits, posts_logits, deters, posts = \
                self.policy.model_forward(obs, acts, is_first)
            """model"""
            observation_loss = -po.log_prob(obs)
            reward_loss = -pr.log_prob(rews)
            # KL balancing
            dyn_loss = kl_div(  # prior -> post
                Independent(OneHotCategoricalStraightThrough(logits=posts_logits.detach()), 1),
                Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
            )
            free_nats = torch.full_like(dyn_loss, self.kl_free_nats)
            dyn_loss = torch.maximum(dyn_loss, free_nats)
            repr_loss = kl_div(  # post -> prior
                Independent(OneHotCategoricalStraightThrough(logits=posts_logits), 1),
                Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
            )
            repr_loss = torch.maximum(repr_loss, free_nats)

            if pc is not None and cont is not None:
                continue_loss = self.continue_scale_factor * -pc.log_prob(cont)
            else:
                continue_loss = torch.zeros_like(reward_loss)

            if self.config.harmony:
                repr_loss *= self.kl_representation / (self.kl_representation + self.kl_dynamic)
                dyn_loss *= self.kl_dynamic / (self.kl_representation + self.kl_dynamic)
                kl_loss = dyn_loss + repr_loss
                observation_loss = self.policy.harmonizer_s1(observation_loss)
                reward_loss = self.policy.harmonizer_s2(reward_loss)
                kl_loss = self.policy.harmonizer_s3(kl_loss)
            else:
                repr_loss *= self.kl_representation
                dyn_loss *= self.kl_dynamic
                kl_loss = dyn_loss + repr_loss
                kl_loss *= self.kl_regularizer
            model_loss = (kl_loss + observation_loss + reward_loss + continue_loss).mean()

        self.optimizer['model'].zero_grad()
        self.scaler.scale(model_loss).backward()
        self.scaler.unscale_(self.optimizer['model'])
        adaptive_grad_clip(self.wm_params)
        # if self.config.world_model.clip_gradients is not None:
        #     torch.nn.utils.clip_grad_norm_(self.wm_params, self.config.world_model.clip_gradients)
        self.scaler.step(self.optimizer['model'])
        self.scaler.update()

        """actor"""
        with autocast(dtype=torch.float16):
            out = self.policy.actor_critic_forward(deters, posts, terms)
            objective, discount, entropy = out['for_actor']
            actor_loss = -torch.mean(discount.detach() * (objective + entropy))

        self.optimizer['actor'].zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.optimizer['actor'])
        adaptive_grad_clip(self.policy.actor.parameters())
        # if self.config.actor.clip_gradients is not None:
        #     torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config.actor.clip_gradients)
        self.scaler.step(self.optimizer['actor'])
        self.scaler.update()

        with autocast(dtype=torch.float16):
            """critic"""
            qv, predicted_target_values, lambda_values = out['for_critic']
            critic_loss = -qv.log_prob(lambda_values.detach()) -qv.log_prob(predicted_target_values.detach())
            critic_loss = torch.mean(discount.squeeze(-1).detach() * critic_loss)

        self.optimizer['critic'].zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.optimizer['critic'])
        adaptive_grad_clip(self.policy.critic.parameters())
        # if self.config.critic.clip_gradients is not None:
        #     torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config.critic.clip_gradients)
        self.scaler.step(self.optimizer['critic'])
        self.scaler.update()

        # TODO replay loss

        self.gradient_step += 1
        # if self.gradient_step % 100 == 0:
        #     print(f'gradient_step: {self.gradient_step}')

        info = {
            "model_loss/model_loss": model_loss.item(),
            "model_loss/obs_loss": observation_loss.mean().item(),
            "model_loss/rew_loss": reward_loss.mean().item(),
            "model_loss/continue_loss": continue_loss.mean().item(),
            "model_loss/kl_loss": kl_loss.mean().item(),

            "actor_loss/actor_loss": actor_loss.item(),
            "actor_loss/reinforce_loss": objective.mean().item(),
            "actor_loss/entropy_loss": entropy.mean().item(),

            "critic_loss/critic_loss": critic_loss.item(),
            "critic_loss/lambda_values": lambda_values.mean().item(),

            "step/gradient_step": self.gradient_step
        }
        if self.config.harmony:
            info.update({'harmonizer/s1': self.policy.harmonizer_s1.get_harmony().item()})
            info.update({'harmonizer/s2': self.policy.harmonizer_s2.get_harmony().item()})
            info.update({'harmonizer/s3': self.policy.harmonizer_s3.get_harmony().item()})
        return info


import torch


def adaptive_grad_clip(params, clip_coef: float = 0.3, eps: float = 1e-6):
    """
    对 model 中每个参数张量执行 AGC：
      如果 ||g||_2 > clip_coef * ||w||_2，则将 g 缩放到 (clip_coef * ||w||_2) / ||g||_2。

    只对 dim>=2 的参数（通常是权重矩阵／卷积核）生效，bias 等一维参数保持不裁剪。

    Args:
      model:    包含参数的 nn.Module
      clip_coef:  梯度上限相对于权重范数的比例 α
      eps:       防止除零的最小值
    """
    for param in params:
        if param.grad is None:
            continue
        # 仅裁剪矩阵／卷积核等 dim>=2 的参数
        if param.dim() < 2:
            continue

        # 1. 计算权重和梯度的 L2 范数
        w_norm = param.data.norm(2)
        g_norm = param.grad.data.norm(2)

        # 2. 计算该层允许的最大梯度
        max_norm = torch.clamp(w_norm * clip_coef, min=eps)

        # 3. 若当前梯度超过阈值，则按比例缩放
        if g_norm > max_norm:
            param.grad.data.mul_(max_norm / (g_norm + 1e-16))
