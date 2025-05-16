import torch

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
        # self.optimizer = {
        #     'model': torch.optim.Adam(self.wm_params, config.learning_rate_model),
        #     'actor': torch.optim.Adam(policy.actor.parameters(), config.learning_rate_actor),
        #     'critic': torch.optim.Adam(policy.critic.parameters(), config.learning_rate_critic)
        # }
        self.optimizer1 = DreamerV3Optimizer(params=self.wm_params, lr=config.learning_rate)
        self.optimizer2 = DreamerV3Optimizer(params=policy.actor.parameters(), lr=config.learning_rate)
        self.optimizer3 = DreamerV3Optimizer(params=policy.critic.parameters(), lr=config.learning_rate)
        # AMP GradScaler for float16
        # self.scaler = GradScaler()
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

        # with autocast(dtype=torch.float16):
        po, pr, pc, priors_logits, posts_logits, deters, posts = \
            self.policy.model_forward(obs, acts, is_first)
        """model"""
        observation_loss = -po.log_prob(obs + 0.5)
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

        self.optimizer1.zero_grad()
        model_loss.backward()
        self.optimizer1.step()
        # self.optimizer['model'].zero_grad()
        # model_loss.backward()
        # self.scaler.scale(model_loss).backward()
        # self.scaler.unscale_(self.optimizer['model'])
        # if self.config.world_model.clip_gradients is not None:
        #     torch.nn.utils.clip_grad_norm_(self.wm_params, self.config.world_model.clip_gradients)
        # self.scaler.step(self.optimizer['model'])
        # self.scaler.update()
        # self.optimizer['model'].step()

        """actor"""
        # with autocast(dtype=torch.float16):
        out = self.policy.actor_critic_forward(deters, posts, terms)
        objective, discount, entropy = out['for_actor']
        actor_loss = -torch.mean(discount.detach() * (objective + entropy))

        self.optimizer2.zero_grad()
        actor_loss.backward()
        self.optimizer2.step()
        # self.optimizer['actor'].zero_grad()
        # actor_loss.backward()
        # self.scaler.scale(actor_loss).backward()
        # self.scaler.unscale_(self.optimizer['actor'])
        # if self.config.actor.clip_gradients is not None:
        #     torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config.actor.clip_gradients)
        # self.scaler.step(self.optimizer['actor'])
        # self.scaler.update()
        # self.optimizer['actor'].step()

        # with autocast(dtype=torch.float16):
        """critic"""
        qv, predicted_target_values, lambda_values = out['for_critic']
        critic_loss = -qv.log_prob(lambda_values.detach()) -qv.log_prob(predicted_target_values.detach())
        critic_loss = torch.mean(discount.squeeze(-1).detach() * critic_loss)

        self.optimizer3.zero_grad()
        critic_loss.backward()
        self.optimizer3.step()
        # self.optimizer['critic'].zero_grad()
        # critic_loss.backward()
        # self.scaler.scale(critic_loss).backward()
        # self.scaler.unscale_(self.optimizer['critic'])
        # if self.config.critic.clip_gradients is not None:
        #     torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config.critic.clip_gradients)
        # self.scaler.step(self.optimizer['critic'])
        # self.scaler.update()
        # self.optimizer['critic'].step()

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
import torch.optim as optim
import torch.nn as nn
import math
import re
# TODO
class DreamerV3Optimizer(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=4e-5,
        agc_clip=0.3, agc_pmin=1e-3,
        beta1=0.9, beta2=0.999, eps=1e-20,
        weight_decay=0.0, wd_regex=r'.*\.weight$', nesterov=False
    ):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, agc_clip=agc_clip, agc_pmin=agc_pmin,
                        beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, wd_regex=wd_regex,
                        nesterov=nesterov)
        super(DreamerV3Optimizer, self).__init__(params, defaults)

        self.wd_pattern = re.compile(wd_regex)

    def __setstate__(self, state):
        super(DreamerV3Optimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            # 重新编译正则，因为状态字典不能存储编译后的正则对象
            self.wd_pattern = re.compile(group['wd_regex'])

    def step(self, closure=None):
        """
        执行一个优化步骤。

        Args:
            closure (callable, optional): 计算损失并返回梯度的闭包。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Get group hyperparameters
            lr = group['lr']
            agc_clip = group['agc_clip']
            agc_pmin = group['agc_pmin']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            # Note: Learning rate scheduling (warmup/anneal) would typically be
            # applied here by modifying the `lr` variable per step,
            # managed by a separate scheduler object or internal step count.
            # For simplicity, we use a fixed LR in this mock.

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data # Get the gradient data

                # --- Apply transformations sequentially ---
                """(ok)"""
                # 1. Adaptive Gradient Clipping (AGC)
                # Equivalent to optax.clip_by_agc
                if agc_clip > 0:
                    pnorm = torch.linalg.norm(p.data.flatten(), 2)
                    gnorm = torch.linalg.norm(grad.flatten(), 2)
                    max_norm = agc_clip * max(agc_pmin, pnorm.item())
                    scale = gnorm / max_norm
                    grad.mul_(torch.tensor(1.0, device=grad.device) / torch.max(torch.tensor(1.0, device=grad.device), scale))


                # Get parameter state
                state = self.state[p]

                # Initialize state variables if needed
                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) # Momentum buffer (like Adam m)
                    state['v'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) # RMS buffer (like Adam v)

                state['step'] += 1
                step = state['step']
                m = state['m']
                v = state['v']

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                """(ok)"""
                # 2. Scale by RMS (Root Mean Square)
                # Equivalent to optax.scale_by_rms (partial: updates v and scales grad)
                # This calculates v and applies v-based scaling
                # v_hat = [beta2 * v + (1 - beta2) * g^2] / (1 - beta2 ** step)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # Update v buffer
                v_hat = v / bias_correction2                     # Apply bias correction
                grad = grad / (v_hat.sqrt() + eps)               # Scale gradient by sqrt(v_hat)
                """(ok)"""
                # 3. Scale by Momentum (with Bias Correction)
                # Equivalent to optax.scale_by_momentum (partial: updates m and returns m_hat)
                # This calculates the momentum term m_hat based on the scaled grad
                # m = [beta1 * m + (1 - beta1) * g_hat] / (1 - beta1 ** step)
                m.mul_(beta1).add_(grad, alpha=1 - beta1) # Update m buffer

                """TODO"""
                # Let's follow Optax's scale_by_momentum.update_fn logic:
                # optax.update_moment(updates, mu, beta, 1) for mu
                # optax.bias_correction(mu, beta, step) for mu_hat if not nesterov
                # optax.update_moment(updates, mu, beta, 1) for mu_nesterov
                # optax.bias_correction(mu_nesterov, beta, step) for mu_hat if nesterov
                # PyTorch Adam implements Nesterov as: grad = grad + beta1 * m_buffer (before updating m_buffer)
                # Let's try to strictly follow Optax formulation for m_hat:
                m_hat_no_nesterov = m / bias_correction1 # m_hat without nesterov
                if nesterov:
                     m_nesterov = m * beta1 + grad * (1-beta1) # This is how Optax's scale_by_momentum seems to compute the Nesterov-like intermediate
                     m_hat = m_nesterov / bias_correction1 # Apply bias correction to the nesterov intermediate
                else:
                    m_hat = m_hat_no_nesterov # Standard momentum with bias correction

                # The result after AGC, RMS scaling, and Momentum is the 'update direction'
                update_direction = m_hat

                """TODO"""
                # 4. Add Weight Decay
                # Equivalent to optax.add_decayed_weights.
                # Optax adds wd * param to the *updates* (after other transformations but before LR).
                # We apply it here to the 'update_direction'
                # Check if weight decay should be applied to THIS parameter
                apply_wd = weight_decay != 0
                if group['wd_regex'] is not None:
                     # Check if the parameter name matches the regex
                     # In torch.optim.Optimizer, we don't have direct access to parameter names here within the loop.
                     # A common workaround is to pass names or split parameters into groups beforehand.
                     # For this mock, we'll assume wd_regex is handled by the group structure
                     # (e.g., creating groups where only 'kernel' or 'weight' params have wd > 0)
                     # Or, more directly in torch, filter `params` list in __init__ based on name & regex.
                     # Let's assume the group['params'] already filtered for wd.
                     pass # If filtering is done upstream or via specific groups

                # Following Optax: updates = f_N(...f_1(grads)...) + wd * param
                # Here f_1 to f_3 are AGC, RMS, Momentum. Let update_intermediate = f_3(f_2(f_1(grad))) = update_direction
                # Then update_with_wd = update_intermediate + weight_decay * p.data
                # This is the L2 regularization gradient that is added to the primary update direction.
                if apply_wd: # We assume apply_wd check is simplified or done via groups
                    update_with_wd = update_direction.add(p.data, alpha=weight_decay)
                else:
                    update_with_wd = update_direction


                # 5. Scale by Learning Rate
                # Equivalent to optax.scale_by_learning_rate
                # updates = lr * update_with_wd
                updates = update_with_wd.mul(lr)


                # Finally, apply updates to the parameter
                # p.data = p.data - updates OR p.data.add_(updates, alpha=-1)
                p.data.add_(updates, alpha=-1)

        return loss # Return the computed loss if closure was provided

