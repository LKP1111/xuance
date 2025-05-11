import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    TanhTransform,
    TransformedDistribution,
)
from torch.distributions.utils import probs_to_logits

from xuance.torch.utils.layers4dreamder import (
    CNN,
    MLP,
    DeCNN,
    LayerNorm,
    LayerNormChannelLast,
    LayerNormGRUCell,
    MultiDecoder,
    MultiEncoder,
    ModuleType,
    cnn_forward, RMSNorm, RMSNormChannelLast, BlockLinear
)
from xuance.torch.utils import sym_log, dotdict, init_weights, uniform_init_weights, compute_stochastic_state
from xuance.torch.utils.operations import trunc_normal_init_weights


class Encoder(nn.Module):
    def __init__(
            self,
            # input_obs_shape
            obs_shape: Sequence[int],
            # encoder
            depth: int = 64,
            mults: Sequence[int] = [2, 3, 4, 4],
            kernel: int = 5,
            stride: int = 1,
            padding: Union[str, int] = 'same',
            layers: int = 3,
            symlog: bool = True,
            dense_units: int = 1024,
            outscale: float = 1.0,
            # others
            pixel: bool = True):
        super().__init__()
        # store for forward
        self.obs_shape = obs_shape
        self.symlog = symlog
        self.pixel = pixel
        # net_init
        li = []
        if not pixel:
            # <=2d vec_obs
            assert len(obs_shape) <= 2
            in_dim = int(np.prod(obs_shape))
            for _ in range(layers):
                li.append(nn.Linear(in_dim, dense_units))
                li.append(RMSNorm(dense_units))
                li.append(nn.SiLU())
                in_dim = dense_units
        else:
            # 3d img_obs
            assert len(obs_shape) == 3
            # channel first, gray or rgb
            assert obs_shape[0] == 1 or obs_shape[0] == 3
            in_dim = obs_shape[0]
            depths = (np.array(mults) * depth).tolist()
            for d in depths:
                li.append(nn.Conv2d(in_dim, d,
                                    kernel_size=kernel,
                                    stride=stride,
                                    padding=padding))
                # v3_official_new: x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
                li.append(nn.MaxPool2d(2, 2))
                li.append(RMSNormChannelLast(d))
                li.append(nn.SiLU())
                in_dim = d
        self.model = nn.Sequential(*li)
        # wb_init
        self.apply(trunc_normal_init_weights(scale=outscale))

    def forward(self, obs: Tensor):
        # [~, 3, 64, 64] -> [B, 3, 64, 64]
        batch_shape = obs.shape[:len(obs.shape) - len(self.obs_shape)]
        obs = obs.view(-1, *self.obs_shape)
        # sym_log & flatten for vec_obs
        get_obs = sym_log if self.symlog and not self.pixel else lambda x: x
        if not self.pixel and len(self.obs_shape) != 1:
            obs = obs.flatten(len(batch_shape), -1)
        out = self.model(get_obs(obs))
        return out.reshape(*batch_shape, -1)  # [~, -1]



class Decoder(nn.Module):
    def __init__(
            self,
            # input_latent_size: deter & stoch; output_obs_shape
            deter_size: int,
            stoch_size: int,
            obs_shape: Sequence[int],
            # encoder
            depth: int = 64,
            mults: Sequence[int] = [2, 3, 4, 4],
            kernel: int = 5,
            stride: int = 1,
            padding: Union[str, int] = 'same',
            blocks: int = 8,
            layers: int = 3,
            dense_units: int = 1024,
            outscale: float = 1.0,
            # others
            pixel: bool = True):
        super().__init__()
        # store for forward
        self.obs_shape = obs_shape
        self.pixel = pixel
        # net_init
        li = []
        if not pixel:
            # <=2d vec_obs
            assert len(obs_shape) <= 2
            in_dim = deter_size + stoch_size
            for _ in range(layers):
                li.append(nn.Linear(in_dim, dense_units))
                li.append(RMSNorm(dense_units))
                li.append(nn.SiLU())
                in_dim = dense_units
            li.append(nn.Linear(dense_units, int(np.prod(obs_shape))))
        else:
            # 3d img_obs
            assert len(obs_shape) == 3
            # channel first, gray or rgb
            assert obs_shape[0] == 1 or obs_shape[0] == 3
            # calc the input shape of Conv2d: [256, 4, 4]
            depths = (np.array(mults) * depth).tolist()
            factor = 2 ** len(depths)
            conv_in = [obs_shape[-1] // factor] * 2
            self.shape = [depths[-1], *conv_in]
            # u: Conv2d input size; g: blocks
            u, g = int(np.prod(self.shape)), blocks  # u = 4096, g = 8
            # deter -> block linear -> shape of convt_in
            self.bl = BlockLinear(deter_size, u, g)
            # stoch -> linear -> shape of convt_in
            self.l = nn.Sequential(
                nn.Linear(stoch_size, dense_units * 2),
                RMSNorm(dense_units * 2), nn.SiLU(),
                nn.Linear(dense_units * 2, u)
            )
            # norm & act after deter + stoch
            self.norm_act = nn.Sequential(RMSNormChannelLast(depths[-1]), nn.SiLU())
            # conv
            in_channel = self.shape[0]
            for d in reversed([obs_shape[0], ] + depths[:-1]):
                li.append(nn.Conv2d(in_channel, d,
                                    kernel_size=kernel,
                                    stride=stride,
                                    padding=padding))
                # v3_official_new: x = x.repeat(2, -2).repeat(2, -3)
                li.append(nn.UpsamplingNearest2d(scale_factor=2))  # unsampling last 2 dimensions
                li.append(RMSNormChannelLast(d))
                li.append(nn.SiLU())
                in_channel = d
        self.model = nn.Sequential(*li)
        # wb_init
        self.apply(trunc_normal_init_weights(scale=outscale))

    def forward(self, deter, stoch):
        batch_shape = deter.shape[:-1]
        x0, x1 = deter, stoch  # x0: [1, 64, 8192]; x1: [1, 64, 32, 64]
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))  # [64, 8192]
        x1 = x1.reshape((-1, x1.shape[-1]))  # [64, 2048]
        if not self.pixel:
            obs = self.model(torch.cat([x0, x1], dim=-1))
        else:
            # deter -> block linear -> shape of conv_in
            x0 = self.bl(x0)  # -> [64, 4096]
            x0 = x0.view(-1, *self.shape)
            # stoch -> linear -> shape of conv_in
            x1 = self.l(x1).view(-1, *self.shape)
            x = self.norm_act(x0 + x1)
            # convt
            obs = self.model(x)
        return obs.reshape(*batch_shape, *self.obs_shape)  # [~, *obs_shape]


class RecurrentModel(nn.Module):
    """Recurrent model for the model-base Dreamer-V3 agent.
    This implementation uses the `models.LayerNormGRUCell`, which combines
    the standard GRUCell from PyTorch with the `nn.LayerNorm`, where the normalization is applied
    right after having computed the projection from the input to the weight space.

    Args:
        input_size (int): the input size of the model.
        dense_units (int): the number of dense units.
        deter_size (int): the size of the recurrent state.
        activation_fn (nn.Module): the activation function.
            Default to SiLU.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
    """

    def __init__(
            self,
            input_size: int,
            deter_size: int,
            dense_units: int,
            activation_fn: nn.Module = nn.SiLU,
            layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
            layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dims=input_size,
            output_dim=None,
            hidden_sizes=[dense_units],
            activation=activation_fn,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=[layer_norm_cls],
            norm_args=[{**layer_norm_kw, "normalized_shape": dense_units}],
        )
        self.rnn = LayerNormGRUCell(
            dense_units,
            deter_size,
            bias=False,
            batch_first=False,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
        )
        self.deter_size = deter_size

    def forward(self, input: Tensor, recurrent_state: Tensor) -> Tensor:
        """
        Compute the next recurrent state from the latent state (stochastic and recurrent states) and the actions.

        Args:
            input (Tensor): the input tensor composed by the stochastic state and the actions concatenated together.
            recurrent_state (Tensor): the previous recurrent state.

        Returns:
            the computed recurrent output and recurrent state.
        """
        feat = self.mlp(input)
        out = self.rnn(feat, recurrent_state)
        return out


class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in
            [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a
            multi-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in
            [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multi-layer perceptron to predict the stochastic part of the latent state.
        distribution_config (Dict[str, Any]): the configs of the distributions.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over states, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`, where `unif = `1 / self.discrete`.
            Defaults to 0.01.
    """

    def __init__(
            self,
            recurrent_model: RecurrentModel,
            representation_model: nn.Module,
            transition_model: nn.Module,
            distribution_config: Dict[str, Any],
            discrete: int = 32,
            unimix: float = 0.01,
            learnable_initial_recurrent_state: bool = True,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.distribution_config = distribution_config
        self.discrete = discrete
        self.unimix = unimix
        if learnable_initial_recurrent_state:
            self.initial_recurrent_state = nn.Parameter(
                torch.zeros(recurrent_model.deter_size, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initial_recurrent_state", torch.zeros(recurrent_model.deter_size, dtype=torch.float32)
            )

    def get_initial_states(self, batch_shape: Union[Sequence[int], torch.Size]) -> Tuple[Tensor, Tensor]:
        initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
        initial_posterior = self._transition(initial_recurrent_state, sample_state=False)[1]
        return initial_recurrent_state, initial_posterior

    def dynamic(
            self, posterior: Tensor, recurrent_state: Tensor, action: Tensor, embedded_obs: Tensor, is_first: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Perform one step of the dynamic learning:
            Recurrent model: compute the recurrent state from the previous latent space, the action taken by the agent,
                i.e., it computes the deterministic state (or ht).
            Transition model: predict the prior from the recurrent output.
            Representation model: compute the posterior from the recurrent state and from
                the embedded observations provided by the environment.
        For more information see [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
        and [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

        Args:
            posterior (Tensor): the stochastic state computed by the representation model (posterior). It is expected
                to be of dimension `[stoch_size, self.discrete]`, which by default is `[32, 32]`.
            recurrent_state (Tensor): a tuple representing the recurrent state of the recurrent model.
            action (Tensor): the action taken by the agent.
            embedded_obs (Tensor): the embedded observations provided by the environment.
            is_first (Tensor): if this is the first step in the episode.

        Returns:
            The recurrent state (Tensor): the recurrent state of the recurrent model.
            The posterior stochastic state (Tensor): computed by the representation model
            The prior stochastic state (Tensor): computed by the transition model
            The logits of the posterior state (Tensor): computed by the transition model from the recurrent state.
            The logits of the prior state (Tensor): computed by the transition model from the recurrent state.
            from the recurrent state and the embbedded observation.
        """
        action = (1 - is_first) * action

        initial_recurrent_state, initial_posterior = self.get_initial_states(recurrent_state.shape[:2])
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        prior_logits, prior = self._transition(recurrent_state)
        posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)
        return recurrent_state, posterior, prior, posterior_logits, prior_logits

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

    def _representation(self, recurrent_state: Tensor, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_state (Tensor): the recurrent state of the recurrent model, i.e.,
                what is called h or deterministic state in
                [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        logits: Tensor = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)

    def _transition(self, recurrent_out: Tensor, sample_state=True) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.
            sampler_state (bool): whether or not to sample the stochastic state.
                Default to True

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """
        logits: Tensor = self.transition_model(recurrent_out)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete, sample=sample_state)

    def imagination(self, prior: Tensor, recurrent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        One-step imagination of the next latent state.
        It can be used several times to imagine trajectories in the latent space (Transition Model).

        Args:
            prior (Tensor): the prior state.
            recurrent_state (Tensor): the recurrent state of the recurrent model.
            actions (Tensor): the actions taken by the agent.

        Returns:
            The imagined prior state (Tuple[Tensor, Tensor]): the imagined prior state.
            The recurrent state (Tensor).
        """
        recurrent_state = self.recurrent_model(torch.cat((prior, actions), -1), recurrent_state)
        _, imagined_prior = self._transition(recurrent_state)
        return imagined_prior, recurrent_state


class Actor(nn.Module):
    def __init__(
            self,
            latent_size: int,
            act_shape: Sequence[int],
            is_continuous: bool,
            # config
            layers: int = 3,
            dense_units: int = 1024,
            init_std: float = 2.0,
            min_std: float = 0.1,
            max_std: float = 1.0,
            unimix: float = 0.01,
            outscale: float = 0.01,
    ):
        super().__init__()
        # store for forward
        self.latent_size = latent_size
        self.act_shape = act_shape
        self.is_continuous = is_continuous
        self.unimix = unimix
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        # net init
        li, in_dim, dense_units = [], latent_size, dense_units
        for _ in range(layers):
            li, in_dim = li + [
                nn.Linear(in_dim, dense_units),
                RMSNorm(dense_units),
                nn.SiLU()
            ], dense_units
        out_dim = np.prod(act_shape) * (2 if is_continuous else 1)
        li.append(nn.Linear(dense_units, out_dim))
        self.model = nn.Sequential(*li)
        # wb_init
        self.apply(trunc_normal_init_weights(scale=outscale))

    def forward(self, x, sample=True):
        x = self.model(x)
        if self.is_continuous:
            # bounded normal
            mean, std = torch.chunk(x, 2, -1)
            std = (self.max_std - self.min_std) * torch.sigmoid(std + self.init_std) + self.min_std
            dist = Independent(Normal(torch.tanh(mean), std), 1)
        else:
            # categorical
            dist = OneHotCategoricalStraightThrough(logits=self._uniform_mix(x))
        act = dist.rsample() if sample else dist.mode
        return act, dist

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        return logits


class Critic(nn.Module):
    def __init__(
            self,
            # input
            latent_size: int,
            # config
            layers: int = 3,
            dense_units: int = 1024,
            bins: int = 255,
            outscale: float = 0.0,
    ) -> None:
        super().__init__()
        # store for forward
        self.latent_size = latent_size
        # net init
        li, in_dim, dense_units = [], latent_size, dense_units
        for _ in range(layers):
            li, in_dim = li + [
                nn.Linear(in_dim, dense_units),
                RMSNorm(dense_units),
                nn.SiLU()
            ], dense_units
        li.append(nn.Linear(dense_units, bins))
        self.model = nn.Sequential(*li)
        # wb_init
        self.apply(trunc_normal_init_weights(scale=outscale))

    def forward(self, x):
        return self.model(x)

class PlayerDV3(nn.Module):
    """
    The model of the Dreamer_v3 player.

    Args:
        encoder (MultiEncoder): the encoder.
        rssm (RSSM): the RSSM model.
        actor (Module): the actor.
        act_shape (Sequence[int]): the dimension of the actions.
        num_envs (int): the number of environments.
        stochastic_size (int): the size of the stochastic state.
        deter_size (int): the size of the recurrent state.
        transition_model (Module): the transition model.
        discrete_size (int): the dimension of a single Categorical variable in the
            stochastic state (prior or posterior).
            Defaults to 32.
        actor_type (str, optional): which actor the player is using ('task' or 'exploration').
            Default to None.
         (bool, optional): whether to use the DecoupledRSSM model.
    """

    def __init__(
            self,
            encoder: MultiEncoder,
            rssm: RSSM,
            actor: Actor,
            act_shape: Sequence[int],
            num_envs: int,
            stochastic_size: int,
            deter_size: int,
            device: torch.device,
            discrete_size: int = 32,
            actor_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.actor = actor
        self.act_shape = act_shape
        self.num_envs = num_envs
        self.stochastic_size = stochastic_size
        self.deter_size = deter_size
        self.device = device
        self.discrete_size = discrete_size
        self.actor_type = actor_type

        self.actions, self.recurrent_state, self.stochastic_state = [None] * 3

    @torch.no_grad()
    def init_states(self,
                    reset_envs: Optional[Sequence[int]] = None,
                    num_envs: Optional[int] = None) -> None:
        """Initialize the states and the actions for the ended environments.

        Args:
            reset_envs (Optional[Sequence[int]], optional): which environments' states to reset.
                If None, then all environments' states are reset.
                Defaults to None.
            num_envs (Optional[int]): the number of environments.
                If None, then it will be self.num_envs  # prop added to deal with xuance test
        """
        num_envs = num_envs if num_envs else self.num_envs  # added to deal with xuance test
        if reset_envs is None or len(reset_envs) == 0:  # reset all
            self.actions = torch.zeros(1, num_envs, np.sum(self.act_shape), device=self.device)
            self.recurrent_state, stochastic_state = self.rssm.get_initial_states((1, num_envs))
            self.stochastic_state = stochastic_state.reshape(1, num_envs, -1)
        else:
            self.actions[:, reset_envs] = torch.zeros_like(self.actions[:, reset_envs])
            self.recurrent_state[:, reset_envs], stochastic_state = self.rssm.get_initial_states((1, len(reset_envs)))
            self.stochastic_state[:, reset_envs] = stochastic_state.reshape(1, len(reset_envs), -1)

    def get_actions(
            self,
            obs: Dict[str, Tensor],
            greedy: bool = False,
            mask: Optional[Dict[str, Tensor]] = None,
    ) -> Sequence[Tensor]:
        """
        Return the greedy actions.

        Args:
            obs (Dict[str, Tensor]): the current observations.
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Optional[Dict[str, Tensor]]): action mask
        Returns:
            The actions the agent has to perform.
        """
        embedded_obs = self.encoder(obs)
        self.recurrent_state = self.rssm.recurrent_model(  # -> h1: [1: batch, 4: envs, 512]
            torch.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )
        _, self.stochastic_state = self.rssm._representation(self.recurrent_state, embedded_obs)
        self.stochastic_state = self.stochastic_state.view(
            *self.stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions, _ = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1).squeeze(0), sample=not greedy)
        return actions


class RewardPredictor(nn.Module):
    def __init__(
            self,
            # input
            latent_size: int,
            # config
            bins: int = 255,
            layers: int = 1,
            dense_units: int = 1024,
            outscale: float = 0.0,
    ) -> None:
        super().__init__()
        # store for forward
        self.latent_size = latent_size
        # net init
        li, in_dim, dense_units = [], latent_size, dense_units
        for _ in range(layers):
            li, in_dim = li + [
                nn.Linear(in_dim, dense_units),
                RMSNorm(dense_units),
                nn.SiLU()
            ], dense_units
        li.append(nn.Linear(dense_units, bins))
        self.model = nn.Sequential(*li)
        # wb_init
        self.apply(trunc_normal_init_weights(scale=outscale))

    def forward(self, x):
        return self.model(x)


class DiscountPredictor(nn.Module):
    def __init__(
            self,
            # input
            latent_size: int,
            # config
            layers: int = 1,
            dense_units: int = 1024,
            outscale: float = 1.0,
    ) -> None:
        super().__init__()
        # store for forward
        self.latent_size = latent_size
        # net init
        li, in_dim, dense_units = [], latent_size, dense_units
        for _ in range(layers):
            li, in_dim = li + [
                nn.Linear(in_dim, dense_units),
                RMSNorm(dense_units),
                nn.SiLU()
            ], dense_units
        li.append(nn.Linear(dense_units, 1))
        self.model = nn.Sequential(*li)
        # wb_init
        self.apply(trunc_normal_init_weights(scale=outscale))

    def forward(self, x):
        return self.model(x)
    

class WorldModel(nn.Module):
    """
    Wrapper class for the World model.

    Args:
        encoder (Module): the encoder.
        rssm (RSSM): the rssm.
        decoder (Module): the observation model.
        reward_predictor (Module): the reward model.
        discount_predictor (Module, optional): the continue model.
    """

    def __init__(
            self,
            encoder,
            rssm: RSSM,
            decoder,
            reward_predictor,
            discount_predictor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.reward_predictor = reward_predictor
        self.discount_predictor = discount_predictor


class DreamerV3WorldModel(nn.Module):
    def __init__(self, act_shape: Sequence[int],
                 is_continuous: bool,
                 config: Dict[str, Any],
                 obs_space: gym.spaces.Dict):
        super().__init__()
        self.act_shape = act_shape
        self.is_continuous = is_continuous
        self.config = config
        self.obs_space = obs_space
        """
        for policy: world_model, actor, critic, target_critic
        for agent: player (link to policy.world_model.~ & policy.actor)
        """
        if self.config.pixel:
            self.obs_space = gym.spaces.Box(0, 255, ((self.obs_space.shape[2], ) + self.obs_space.shape[:2]), np.uint8)
        self.world_model, self.actor, self.critic, self.target_critic, self.player = (
            DreamerV3WorldModel._build_model(
                self.act_shape,
                self.is_continuous,
                self.config,
                self.obs_space
            )
        )

    @staticmethod
    def _build_model(
        act_shape: Sequence[int],
        is_continuous: bool,
        config: Dict[str, Any],
        obs_space: gym.spaces.Dict,
    ) -> Tuple[WorldModel, Actor, nn.Module, nn.Module, PlayerDV3]:
        # world_model, actor, critic, target_critic, player
        """Build the models

        Args:
            act_shape (Sequence[int]): the dimension of the actions.
            is_continuous (bool): whether or not the actions are continuous.
            config (DictConfig): the configs of DreamerV3.
            obs_space (Dict[str, Any]): the observation space.

        Returns:
            The world model (WorldModel): composed by the encoder, rssm, observation and
            reward models and the continue model.
            The actor (nn.Module).
            The critic (nn.Module).
            The target critic (nn.Module).
        """

        config = dotdict(vars(config))
        world_model_config = config.world_model
        """deter_size; stoch_size; model_state_size"""
        # Sizes
        deter_size = world_model_config.recurrent_model.deter_size
        stochastic_size = world_model_config.stochastic_size * world_model_config.discrete_size
        latent_size = stochastic_size + deter_size
        # Define models
        obs_shape = obs_space.shape
        wm_config = config.world_model
        encoder = Encoder(obs_shape=obs_shape, pixel=config.pixel, **wm_config.encoder).to(config.device)
        decoder = Decoder(
            deter_size=(t := wm_config.rssm)['deter_size'],
            stoch_size=t['stoch_size'] * t['classes'],
            obs_shape=obs_shape, pixel=config.pixel,
            **wm_config.decoder
        ).to(config.device)
        # TODO start
        recurrent_model = RecurrentModel(
            input_size=int(sum(act_shape) + stochastic_size),
            deter_size=world_model_config.recurrent_model.deter_size,
            dense_units=world_model_config.recurrent_model.dense_units,
            layer_norm_cls=LayerNorm,
            layer_norm_kw=world_model_config.recurrent_model.layer_norm.kw,
        )
        represention_model_input_size = encoder(torch.zeros(1, *obs_shape).to(config.device)).shape[-1]
        represention_model_input_size += deter_size
        representation_ln_cls = LayerNorm
        representation_model = MLP(
            input_dims=represention_model_input_size,  # (h1, x1) -> z1
            output_dim=stochastic_size,
            hidden_sizes=[world_model_config.representation_model.hidden_size],
            activation=nn.SiLU,
            layer_args={"bias": representation_ln_cls == nn.Identity},
            flatten_dim=None,
            norm_layer=[representation_ln_cls],
            norm_args=[
                {
                    **world_model_config.representation_model.layer_norm.kw,
                    "normalized_shape": world_model_config.representation_model.hidden_size,
                }
            ],
        )
        transition_ln_cls = LayerNorm
        transition_model = MLP(
            input_dims=deter_size,
            output_dim=stochastic_size,
            hidden_sizes=[world_model_config.transition_model.hidden_size],
            activation=nn.SiLU,
            layer_args={"bias": transition_ln_cls == nn.Identity},
            flatten_dim=None,
            norm_layer=[transition_ln_cls],
            norm_args=[
                {
                    **world_model_config.transition_model.layer_norm.kw,
                    "normalized_shape": world_model_config.transition_model.hidden_size,
                }
            ],
        )

        rssm_cls = RSSM
        rssm = rssm_cls(
            recurrent_model=recurrent_model.apply(init_weights),
            representation_model=representation_model.apply(init_weights),
            transition_model=transition_model.apply(init_weights),
            distribution_config=config.distribution,
            discrete=world_model_config.discrete_size,
            unimix=config.unimix,
            learnable_initial_recurrent_state=config.world_model.learnable_initial_recurrent_state,
        ).to(config.device)
        # TODO end
        reward_predictor = RewardPredictor(latent_size=latent_size, **wm_config.reward_predictor).to(config.device)
        discount_predictor = DiscountPredictor(latent_size=latent_size, **wm_config.discount_predictor).to(config.device)
        # reward_ln_cls = LayerNorm
        # reward_predictor = MLP(
        #     input_dims=latent_size,
        #     output_dim=world_model_config.reward_predictor.bins,
        #     hidden_sizes=[world_model_config.reward_predictor.dense_units] * world_model_config.reward_predictor.mlp_layers,
        #     activation=nn.SiLU,
        #     layer_args={"bias": reward_ln_cls == nn.Identity},
        #     flatten_dim=None,
        #     norm_layer=reward_ln_cls,
        #     norm_args={
        #         **world_model_config.reward_predictor.layer_norm.kw,
        #         "normalized_shape": world_model_config.reward_predictor.dense_units,
        #     },
        # ).to(config.device)
        #
        # discount_ln_cls = LayerNorm
        # discount_predictor = MLP(
        #     input_dims=latent_size,
        #     output_dim=1,
        #     hidden_sizes=[world_model_config.discount_predictor.dense_units] * world_model_config.discount_predictor.mlp_layers,
        #     activation=nn.SiLU,
        #     layer_args={"bias": discount_ln_cls == nn.Identity},
        #     flatten_dim=None,
        #     norm_layer=discount_ln_cls,
        #     norm_args={
        #         **world_model_config.discount_predictor.layer_norm.kw,
        #         "normalized_shape": world_model_config.discount_predictor.dense_units,
        #     },
        # ).to(config.device)



        world_model = WorldModel(
            encoder,
            rssm,
            decoder,
            reward_predictor,
            discount_predictor,
        )

        
        actor = Actor(latent_size=latent_size, act_shape=act_shape,
                      is_continuous=is_continuous, **config.actor).to(config.device)
        critic = Critic(latent_size=latent_size, **config.critic).to(config.device)

        if config.hafner_initialization:
            # actor.mlp_heads.apply(uniform_init_weights(1.0))
            # critic.model[-1].apply(uniform_init_weights(0.0))
            rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
            rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))
            # world_model.reward_predictor.model[-1].apply(uniform_init_weights(0.0))
            # world_model.discount_predictor.model[-1].apply(uniform_init_weights(1.0))
            # if mlp_decoder is not None:
            #     mlp_decoder.heads.apply(uniform_init_weights(1.0))
            # if cnn_decoder is not None:
            #     cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))

        player = PlayerDV3(  # encoder, rssm, actor
            copy.deepcopy(world_model.encoder),
            copy.deepcopy(world_model.rssm),
            copy.deepcopy(actor),
            act_shape,
            config.parallels,
            config.world_model.stochastic_size,
            config.world_model.recurrent_model.deter_size,
            config.device,
            discrete_size=config.world_model.discrete_size,
        )
        target_critic = copy.deepcopy(critic)
        # Tie weights between the agent and the player
        for agent_p, p in zip(world_model.encoder.parameters(), player.encoder.parameters()):
            p.data = agent_p.data
        for agent_p, p in zip(world_model.rssm.parameters(), player.rssm.parameters()):
            p.data = agent_p.data
        for agent_p, p in zip(actor.parameters(), player.actor.parameters()):
            p.data = agent_p.data
        return world_model, actor, critic, target_critic, player
