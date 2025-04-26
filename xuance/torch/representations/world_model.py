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
    BlockLinear,
    MultiDecoder,
    MultiEncoder,
    RMSNormChannelLast,
    RMSNorm,
    cnn_forward
)
from xuance.torch.utils import (
    ModuleType,
    sym_log, 
    dotdict, 
    init_weights, 
    uniform_init_weights, 
    compute_stochastic_state, 
)
from xuance.torch.utils.operations import trunc_normal_init_weights


class Encoder(nn.Module):
    def __init__(
            self,
            # input_obs_shape
            obs_shape: Sequence[int],
            # encoder
            depth: int =  64,
            mults: Sequence[int] = [2, 3, 4, 4],
            kernel: int = 5,
            stride: int = 1,
            padding: Union[str, int] = 'same',
            layers: int = 3,
            symlog: bool = True,
            dense_units: int = 1024,
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

# # encoder test (ok)
# encoder = Encoder((3, 64, 64)).to("cuda:1").apply(trunc_normal_init_weights())
# out = encoder(torch.zeros(16, 64, 3, 64, 64).to("cuda:1")).shape  # [16, 64, 4096]

# encoder = Encoder((4, ), pixel=False).to("cuda:1").apply(trunc_normal_init_weights())
# out = encoder(torch.zeros(16, 64, 4).to("cuda:1")).shape  # [16, 64, 1024]

class Decoder(nn.Module):
    def __init__(
            self,
            # input_latent_size: deter + stoch; output_obs_shape
            deter_size: int,  
            stoch_size: int,  
            obs_shape: Sequence[int],
            # encoder
            depth: int =  64,
            mults: Sequence[int] = [2, 3, 4, 4],
            kernel: int = 5,
            stride: int = 1,
            padding: Union[str, int] = 'same',
            blocks: int = 8,
            layers: int = 3,
            dense_units: int = 1024,
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
            for d in reversed([obs_shape[0],] + depths[:-1]):
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

# # decoder test (ok)
# decoder = Decoder(8192, 2048, (3, 64, 64)).to("cuda:1").apply(trunc_normal_init_weights())
# out = decoder(torch.zeros(16, 64, 8192).to("cuda:1"), 
#               torch.zeros(16, 64, 64, 32).to("cuda:1")).shape  # [16, 64, 3, 64, 64]

# decoder = Decoder(8192, 2048, (4,), pixel=False).to("cuda:1").apply(trunc_normal_init_weights())
# out = decoder(torch.zeros(16, 64, 8192).to("cuda:1"), 
#               torch.zeros(16, 64, 64, 32).to("cuda:1")).shape  # [16, 64, 4]



# class CNNEncoder(nn.Module):
#     """The Dreamer-V3 image encoder. This is composed of 4 `nn.Conv2d` with
#     kernel_size=3, stride=2 and padding=1. No bias is used if a `nn.Norm`
#     is used after the convolution. This 4-stages model assumes that the image
#     is a 64x64 and it ends with a resolution of 4x4. If more than one image is to be encoded, then those will
#     be concatenated on the channel dimension and fed to the encoder.

#     Args:
#         input_channels (Sequence[int]): the input channels, one for each image observation to encode.
#         image_size (Tuple[int, int]): the image size as (Height,Width).
#         channels_multiplier (int): the multiplier for the output channels. Given the 4 stages, the 4 output channels
#             will be [1, 2, 4, 8] * `channels_multiplier`.
#         norm_cls (Callable[..., nn.Module]): the norm to apply after the input projection.
#             Defaults to RMSNorm.
#         activation (ModuleType, optional): the activation function.
#             Defaults to nn.SiLU.
#         stages (int, optional): how many stages for the CNN.
#     """

#     def __init__(
#             self,
#             input_channels: Sequence[int],
#             image_size: Tuple[int, int],
#             channels_multiplier: int,
#             norm_cls: Callable[..., nn.Module] = RMSNormChannelLast,
#             activation: ModuleType = nn.SiLU,
#             stages: int = 4,
#     ) -> None:
#         super().__init__()
#         self.input_dim = (sum(input_channels), *image_size)
#         self.model = nn.Sequential(
#             CNN(
#                 input_channels=self.input_dim[0],
#                 hidden_channels=(torch.tensor([2 ** i for i in range(stages)]) * channels_multiplier).tolist(),
#                 cnn_layer=nn.Conv2d,
#                 layer_args={"kernel_size": 4, "stride": 2, "padding": 1, "bias": True},
#                 activation=activation,
#                 norm_layer=[norm_cls] * stages,
#                 norm_args=[
#                     {"normalized_shape": (2 ** i) * channels_multiplier} for i in range(stages)
#                 ],
#             ),
#             nn.Flatten(-3, -1),
#         )
#         with torch.no_grad():
#             self.output_dim = self.model(torch.zeros(1, *self.input_dim)).shape[-1]

#     def forward(self, obs: Tensor) -> Tensor:
#         return cnn_forward(self.model, obs, obs.shape[-3:], (-1,))


# class CNNDecoder(nn.Module):
#     """The exact inverse of the `CNNEncoder` class. It assumes an initial resolution
#     of 4x4, and in 4 stages reconstructs the observation image to 64x64. If multiple
#     images are to be reconstructed, then it will create a dictionary with an entry
#     for every reconstructed image. No bias is used if a `nn.Norm` is used after
#     the `nn.Conv2dTranspose` layer.

#     Args:
#         output_channels (Sequence[int]): the output channels, one for every image observation.
#         channels_multiplier (int): the channels multiplier, same for the encoder network.
#         latent_state_size (int): the size of the latent state. Before applying the decoder,
#             a `nn.Linear` layer is used to project the latent state to a feature vector
#             of dimension [8 * `channels_multiplier`, 4, 4].
#         cnn_encoder_output_dim (int): the output of the image encoder. It should be equal to
#             8 * `channels_multiplier` * 4 * 4.
#         image_size (Tuple[int, int]): the final image size.
#         activation (nn.Module, optional): the activation function.
#             Defaults to nn.SiLU.
#         norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
#             Defaults to NormChannelLast.
#             Default to {"eps": 1e-3}.
#         stages (int): how many stages in the CNN decoder.
#     """

#     def __init__(
#             self,
#             output_channels: Sequence[int],
#             channels_multiplier: int,
#             latent_state_size: int,
#             cnn_encoder_output_dim: int,
#             image_size: Tuple[int, int],
#             activation: nn.Module = nn.SiLU,
#             norm_cls: Callable[..., nn.Module] = RMSNormChannelLast,
#             stages: int = 4,
#     ) -> None:
#         super().__init__()
#         self.output_channels = output_channels
#         self.cnn_encoder_output_dim = cnn_encoder_output_dim
#         self.image_size = image_size
#         self.output_dim = (sum(output_channels), *image_size)
#         self.model = nn.Sequential(
#             nn.Linear(latent_state_size, cnn_encoder_output_dim),
#             nn.Unflatten(1, (-1, 4, 4)),
#             DeCNN(
#                 input_channels=(2 ** (stages - 1)) * channels_multiplier,
#                 hidden_channels=(
#                                         torch.tensor(
#                                             [2 ** i for i in reversed(range(stages - 1))]) * channels_multiplier
#                                 ).tolist()
#                                 + [self.output_dim[0]],
#                 cnn_layer=nn.ConvTranspose2d,
#                 layer_args=[
#                                {"kernel_size": 4, "stride": 2, "padding": 1, "bias": True}
#                                for _ in range(stages - 1)
#                            ]
#                            + [{"kernel_size": 4, "stride": 2, "padding": 1}],
#                 activation=[activation for _ in range(stages - 1)] + [None],
#                 norm_layer=[norm_cls for _ in range(stages - 1)] + [None],
#                 norm_args=[
#                               {"normalized_shape": (2 ** (stages - i - 2)) * channels_multiplier}
#                               for i in range(stages - 1)
#                           ]
#                           + [None],
#             ),
#         )

#     def forward(self, latent_states: Tensor) -> List[Tensor]:
#         cnn_out = cnn_forward(self.model, latent_states, (latent_states.shape[-1],), self.output_dim)
#         return torch.split(cnn_out, self.output_channels, -3)


# class MLPEncoder(nn.Module):
#     """The Dreamer-V3 vector encoder. This is composed of N `nn.Linear` layers, where
#     N is specified by `mlp_layers`. No bias is used if a `nn.Norm` is used after the linear layer.
#     If more than one vector is to be encoded, then those will concatenated on the last
#     dimension before being fed to the encoder.

#     Args:
#         input_dims (Sequence[int]): the dimensions of every vector to encode.
#         mlp_layers (int, optional): how many mlp layers.
#             Defaults to 4.
#         dense_units (int, optional): the dimension of every mlp.
#             Defaults to 512.
#         norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
#             Defaults to Norm.
#         activation (ModuleType, optional): the activation function after every layer.
#             Defaults to nn.SiLU.
#         sym_log_inputs (bool, optional): whether to squash the input with the sym_log function.
#             Defaults to True.
#     """

#     def __init__(
#             self,
#             input_dims: Sequence[int],
#             mlp_layers: int = 4,
#             dense_units: int = 512,
#             norm_cls: Callable[..., nn.Module] = RMSNorm,
#             activation: ModuleType = nn.SiLU,
#             sym_log_inputs: bool = True,
#     ) -> None:
#         super().__init__()
#         self.input_dim = sum(input_dims)  # [4]
#         self.model = MLP(
#             self.input_dim,
#             None,
#             [dense_units] * mlp_layers,
#             activation=activation,
#             layer_args={"bias": True},
#             norm_layer=norm_cls,
#             norm_args={"normalized_shape": dense_units},
#         )
#         self.output_dim = dense_units
#         self.sym_log_inputs = sym_log_inputs  # True

#     def forward(self, obs: Tensor) -> Tensor:
#         # x = torch.cat([sym_log(obs[k]) if self.sym_log_inputs else obs[k] for k in self.keys], -1)
#         return self.model(sym_log(obs))


# class MLPDecoder(nn.Module):
#     """The exact inverse of the MLPEncoder. This is composed of N `nn.Linear` layers, where
#     N is specified by `mlp_layers`. No bias is used if a `nn.Norm` is used after the linear layer.
#     If more than one vector is to be decoded, then it will create a dictionary with an entry
#     for every reconstructed vector.

#     Args:
#         keys (Sequence[str]): the keys representing the vector observations to decode.
#         output_dims (Sequence[int]): the dimensions of every vector to decode.
#         latent_state_size (int): the dimension of the latent state.
#         mlp_layers (int, optional): how many mlp layers.
#             Defaults to 4.
#         dense_units (int, optional): the dimension of every mlp.
#             Defaults to 512.
#         norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
#             Defaults to Norm.
#         activation (ModuleType, optional): the activation function after every layer.
#             Defaults to nn.SiLU.
#     """

#     def __init__(
#         self,
#         output_dims: Sequence[int],
#         latent_state_size: int,
#         mlp_layers: int = 4,
#         dense_units: int = 512,
#         activation: ModuleType = nn.SiLU,
#         norm_cls: Callable[..., nn.Module] = RMSNorm,
#         norm_args = {"normalized_shape": 512}
#     ) -> None:
#         super().__init__()
#         self.output_dims = output_dims
#         self.model = MLP(
#             latent_state_size,
#             None,
#             [dense_units] * mlp_layers,
#             activation=activation,
#             layer_args={"bias": True},
#             norm_layer=norm_cls,
#             norm_args=norm_args
#         )
#         self.heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in self.output_dims])

#     def forward(self, latent_states: Tensor) -> Tensor:
#         x = self.model(latent_states)
#         return self.heads[0](x)  # revised to adapt to xuance


class RecurrentModel(nn.Module):
    """Recurrent model for the model-base Dreamer-V3 agent.
    This implementation uses the `models.BlockLinear`, 
    which enables the sequence model to be a GRU with block-diagonal recurrent weights of 8 blocks,
    to allow for a large number of memory units without quadratic increase in parameters and FLOPs

    Args:
        deter_size (int): the size of the recurrent state.
        stoch_size (int): the size of the stochastic state.
        action_size (int): the size of the action.
        hidden_size (int): the number of dense units.
        blocks (int): the number of blocks.
        dyn_layer (int): the number of dyn_layer (block mlp).
        activation_fn (nn.Module): the activation function.
            Default to SiLU.
        norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to Norm.
    """
    
    def __init__(
            self,
            deter_size: int,
            stoch_size: int,
            action_size: int,
            hidden_size: int,  # related to config.dense_units
            blocks: int=8,
            dyn_layer: int=1, 
            activation_fn: nn.Module = nn.SiLU,
            norm_cls: Callable[..., nn.Module] = RMSNorm,
    ) -> None:
        super().__init__()
        self.g = blocks
        self.hidden_size = hidden_size
        self.deter_size = self.recurrent_state_size = deter_size
        self.stoch_size = stoch_size
        self.action_size = action_size
        self.dyn_layer = dyn_layer
        # 3 linear
        self.linear1 = nn.Linear(self.deter_size, self.hidden_size)
        self.linear2 = nn.Linear(self.stoch_size, self.hidden_size)
        self.linear3 = nn.Linear(self.action_size, self.hidden_size)
        self.norm1 = norm_cls(self.hidden_size)
        self.norm2 = norm_cls(self.hidden_size)
        self.norm3 = norm_cls(self.hidden_size)
        self.activation_fn = activation_fn()
        # block mlp
        self.dyn_li = nn.ModuleList()
        self.norm_li = nn.ModuleList()
        # in: g * hidden * 3 + deter; out: self.deter_size * 3
        in_dim = self.g * self.hidden_size * 3 + self.deter_size
        out_dim = self.deter_size * 3
        for _ in range(self.dyn_layer):
            self.dyn_li.append(BlockLinear(in_dim, self.deter_size, self.g, bias=False))
            self.norm_li.append(norm_cls(self.deter_size))
            in_dim = self.deter_size
        self.dyn_out = BlockLinear(self.deter_size, out_dim, self.g, bias=False)
        

    def forward(self, deter: Tensor, stoch: Tensor, action: Tensor) -> Tensor:
        """
        Compute the next recurrent state from the latent state (stochastic and recurrent states) and the actions.

        Args:
            deter (Tensor): the previous recurrent state.
            stoch (Tensor): the previous stochastic state.
            action (Tensor): action

        Returns:
            the computed recurrent output and recurrent state.
        """
        # seq_batch2batch
        origin_shape = deter.shape[:-1]
        B = np.prod(origin_shape)
        deter = deter.view(B, -1)
        stoch = stoch.view(B, -1)
        action = action.view(B, -1)
        action /= torch.max(torch.as_tensor(1.0), torch.abs(action)).detach()
        g = self.g
        hidden_size = self.hidden_size
        x1 = self.activation_fn(self.norm1(self.linear1(deter)))
        x2 = self.activation_fn(self.norm2(self.linear2(stoch)))
        x3 = self.activation_fn(self.norm3(self.linear3(action)))
        
        # -> [B, g, hidden * 3]
        x = torch.cat([x1, x2, x3], -1).unsqueeze(1).expand(-1, g, hidden_size * 3)
        # -> [B, g, hidden * 3 + deter // g]
        x = torch.cat([x, deter.view(B, g, -1)], -1)
        # -> [B, g * hidden * 3 + deter]
        x = x.view(B, -1)
        # -> [B, deter * 3]
        for i in range(self.dyn_layer):
            x = self.dyn_li[i](x)
            x = self.norm_li[i](x)
        x = self.dyn_out(x)

        reset, cand, update = [y.reshape(B, -1) for y in torch.chunk(x.view(B, g, -1), 3, -1)]
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        next_deter = update * cand + (1 - update) * deter
        return next_deter.view(*origin_shape, -1)


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
                torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initial_recurrent_state", torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
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

        recurrent_state = self.recurrent_model(recurrent_state, posterior, action)
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
        recurrent_state = self.recurrent_model(recurrent_state, prior, actions)
        _, imagined_prior = self._transition(recurrent_state)
        return imagined_prior, recurrent_state


class Actor(nn.Module):
    """
    The wrapper class of the Dreamer_v2 Actor model.

    Args:
        latent_state_size (int): the dimension of the latent state (stochastic size + recurrent_state_size).
        actions_dim (Sequence[int]): the dimension in output of the actor.
            The number of actions if continuous, the dimension of the action if discrete.
        is_continuous (bool): whether or not the actions are continuous.
        distribution_config (Dict[str, Any]): The configs of the distributions.
        init_std (float): the amount to sum to the standard deviation.
            Default to 0.0.
        min_std (float): the minimum standard deviation for the actions.
            Default to 1.0.
        max_std (float): the maximum standard deviation for the actions.
            Default to 1.0.
        dense_units (int): the dimension of the hidden dense layers.
            Default to 1024.
        activation (int): the activation function to apply after the dense layers.
            Default to nn.SiLU.
        mlp_layers (int): the number of dense layers.
            Default to 5.
        norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to RMSNorm.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over actions, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`,
            where `unif = `1 / self.discrete`.
            Defaults to 0.01.
        action_clip (float): the action clip parameter.
            Default to 1.0.
    """

    def __init__(
            self,
            latent_state_size: int,
            actions_dim: Sequence[int],
            is_continuous: bool,
            distribution_config: Dict[str, Any],
            init_std: float = 0.0,
            min_std: float = 1.0,
            max_std: float = 1.0,
            dense_units: int = 1024,
            activation: nn.Module = nn.SiLU,
            mlp_layers: int = 5,
            norm_cls: Callable[..., nn.Module] = RMSNorm,
            unimix: float = 0.01,
            action_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.distribution_config = distribution_config
        self.distribution = distribution_config.get("type", "auto").lower()
        if self.distribution not in ("auto", "normal", "tanh_normal", "discrete", "scaled_normal"):
            raise ValueError(
                "The distribution must be on of: `auto`, `discrete`, `normal`, `tanh_normal` and `scaled_normal`. "
                f"Found: {self.distribution}"
            )
        if self.distribution == "discrete" and is_continuous:
            raise ValueError("You have choose a discrete distribution but `is_continuous` is true")
        if self.distribution == "auto":
            if is_continuous:
                self.distribution = "scaled_normal"
            else:
                self.distribution = "discrete"
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=None,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=activation,
            flatten_dim=None,
            layer_args={"bias": True},
            norm_layer=norm_cls,
            norm_args={
                "normalized_shape": dense_units,
            },
        )
        if is_continuous:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, np.sum(actions_dim) * 2)])
        else:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, action_dim) for action_dim in actions_dim])
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self._unimix = unimix
        self._action_clip = action_clip

    def forward(
            self, state: Tensor, greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Sequence[Distribution]]:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Dict[str, Tensor], optional): the mask to use on the actions.
                Default to None.

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
            The distribution of the actions
        """
        out: Tensor = self.model(state)
        pre_dist: List[Tensor] = [head(out) for head in self.mlp_heads]
        if self.is_continuous:
            mean, std = torch.chunk(pre_dist[0], 2, -1)
            if self.distribution == "tanh_normal":
                mean = 5 * torch.tanh(mean / 5)
                std = F.softplus(std + self.init_std) + self.min_std
                actions_dist = Normal(mean, std)
                actions_dist = Independent(TransformedDistribution(actions_dist, TanhTransform()), 1)
            elif self.distribution == "normal":
                actions_dist = Normal(mean, std)
                actions_dist = Independent(actions_dist, 1)
            elif self.distribution == "scaled_normal":
                std = (self.max_std - self.min_std) * torch.sigmoid(std + self.init_std) + self.min_std
                dist = Normal(torch.tanh(mean), std)
                actions_dist = Independent(dist, 1)
            else:
                actions_dist = None
            if not greedy:
                actions = actions_dist.rsample()
            else:
                actions = actions_dist.mode
            if self._action_clip > 0.0:
                action_clip = torch.full_like(actions, self._action_clip)
                actions = actions * (action_clip / torch.maximum(action_clip, torch.abs(actions))).detach()
            actions = [actions]
            actions_dist = [actions_dist]
        else:
            actions_dist = []
            actions = []
            for logits in pre_dist:
                actions_dist.append(OneHotCategoricalStraightThrough(logits=self._uniform_mix(logits)))
                if not greedy:
                    actions.append(actions_dist[-1].rsample())
                else:
                    actions.append(actions_dist[-1].mode)
        return tuple(actions), tuple(actions_dist)

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        if self._unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - self._unimix) * probs + self._unimix * uniform
            logits = probs_to_logits(probs)
        return logits


class PlayerDV3(nn.Module):
    """
    The model of the Dreamer_v3 player.

    Args:
        encoder (MultiEncoder): the encoder.
        rssm (RSSM): the RSSM model.
        actor (Module): the actor.
        actions_dim (Sequence[int]): the dimension of the actions.
        num_envs (int): the number of environments.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
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
            actions_dim: Sequence[int],
            num_envs: int,
            stochastic_size: int,
            recurrent_state_size: int,
            device: torch.device,
            discrete_size: int = 32,
            actor_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.actor = actor
        self.actions_dim = actions_dim
        self.num_envs = num_envs
        self.stochastic_size = stochastic_size
        self.recurrent_state_size = recurrent_state_size
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
            self.actions = torch.zeros(1, num_envs, np.sum(self.actions_dim), device=self.device)
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
            self.recurrent_state, self.stochastic_state, self.actions
        )
        _, self.stochastic_state = self.rssm._representation(self.recurrent_state, embedded_obs)
        self.stochastic_state = self.stochastic_state.view(
            *self.stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions, _ = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1), greedy, mask)
        self.actions = torch.cat(actions, -1)
        return actions


class WorldModel(nn.Module):
    """
    Wrapper class for the World model.

    Args:
        encoder (Module): the encoder.
        rssm (RSSM): the rssm.
        observation_model (Module): the observation model.
        reward_model (Module): the reward model.
        continue_model (Module, optional): the continue model.
    """

    def __init__(
            self,
            encoder,
            rssm: RSSM,
            observation_model,
            reward_model,
            continue_model,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.continue_model = continue_model


class DreamerV3WorldModel(nn.Module):
    def __init__(self, actions_dim: Sequence[int],
                 is_continuous: bool,
                 config: Dict[str, Any],
                 obs_space: gym.spaces.Dict):
        super().__init__()
        self.actions_dim = actions_dim
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
                self.actions_dim,
                self.is_continuous,
                self.config,
                self.obs_space
            )
        )

    @staticmethod
    def _build_model(
        actions_dim: Sequence[int],
        is_continuous: bool,
        config: Dict[str, Any],
        obs_space: gym.spaces.Dict,
    ) -> Tuple[WorldModel, Actor, nn.Module, nn.Module, PlayerDV3]:
        # world_model, actor, critic, target_critic, player
        """Build the models

        Args:
            actions_dim (Sequence[int]): the dimension of the actions.
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
        actor_config = config.actor
        critic_config = config.critic
        
        """
        v3_official new structure uses bias linear & rms_norm
        differ from v3_official old structure: no bias linear & layer_norm
        """
        assert config.norm == 'rms'  # assert the structure related prop
        norm_cls = RMSNorm
        norm_cls_pixel = RMSNormChannelLast

        """deter_size; stoch_size; model_state_size"""
        # Sizes
        recurrent_state_size = world_model_config.recurrent_model.recurrent_state_size
        stochastic_size = world_model_config.stochastic_size * world_model_config.discrete_size
        latent_state_size = stochastic_size + recurrent_state_size
        # Define models
        cnn_stages = int(np.log2(config.env_config.screen_size) - np.log2(4))  # 4
        cnn_encoder = (
            CNNEncoder(
                input_channels=[int(np.prod(obs_space.shape[:-2]))],
                image_size=obs_space.shape[-2:],
                channels_multiplier=world_model_config.encoder.cnn_channels_multiplier,
                norm_cls=norm_cls_pixel,
                activation=nn.SiLU,
                stages=cnn_stages,
            )
            if config.pixel else None
        )
        mlp_encoder = (
            MLPEncoder(
                input_dims=[obs_space.shape[0]],
                mlp_layers=world_model_config.encoder.mlp_layers,
                dense_units=world_model_config.encoder.dense_units,
                activation=nn.SiLU,
                norm_cls=norm_cls,
            )
            if not config.pixel else None
        )
        encoder = MultiEncoder(cnn_encoder, mlp_encoder).to(config.device)

        recurrent_model = RecurrentModel(
            deter_size=recurrent_state_size,
            stoch_size=stochastic_size,
            action_size=int(sum(actions_dim)),
            hidden_size=world_model_config.recurrent_model.dense_units,
            blocks=world_model_config.recurrent_model.blocks,
            dyn_layer=world_model_config.recurrent_model.dyn_layer, 
            norm_cls=norm_cls,
        )
        # recurrent_model(torch.zeros(16, 512), torch.zeros(16, 512), torch.zeros(16, 4))  # test
        represention_model_input_size = encoder.output_dim
        represention_model_input_size += recurrent_state_size
        representation_model = MLP(
            input_dims=represention_model_input_size,  # (h1, x1) -> z1
            output_dim=stochastic_size,
            hidden_sizes=[world_model_config.representation_model.hidden_size],  # post: obslayers: 1
            activation=nn.SiLU,
            layer_args={"bias": True},
            flatten_dim=None,
            norm_layer=norm_cls,
            norm_args={"normalized_shape": world_model_config.representation_model.hidden_size}
        )
        transition_model = MLP(
            input_dims=recurrent_state_size,
            output_dim=stochastic_size,
            hidden_sizes=[world_model_config.transition_model.hidden_size] * 2,  # prior: imglayers: 2
            activation=nn.SiLU,
            layer_args={"bias": True},
            flatten_dim=None,
            norm_layer=norm_cls,
            norm_args={"normalized_shape": world_model_config.discount_model.dense_units}
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

        cnn_decoder = (
            CNNDecoder(
                output_channels=[int(np.prod(obs_space.shape[:-2]))],
                channels_multiplier=world_model_config.observation_model.cnn_channels_multiplier,
                latent_state_size=latent_state_size,
                cnn_encoder_output_dim=cnn_encoder.output_dim,
                image_size=obs_space.shape[-2:],
                activation=nn.SiLU,
                norm_cls=norm_cls_pixel,
                stages=cnn_stages,
            )
            if config.pixel else None
        )
        mlp_decoder = (
            MLPDecoder(
                output_dims=[obs_space.shape[0]],
                latent_state_size=latent_state_size,
                mlp_layers=world_model_config.observation_model.mlp_layers,
                dense_units=world_model_config.observation_model.dense_units,
                activation=nn.SiLU,
                norm_cls=norm_cls,
                norm_args={"normalized_shape": world_model_config.discount_model.dense_units}
            )
            if not config.pixel else None
        )
        observation_model = MultiDecoder(cnn_decoder, mlp_decoder).to(config.device)

        reward_model = MLP(
            input_dims=latent_state_size,
            output_dim=world_model_config.reward_model.bins,
            hidden_sizes=[world_model_config.reward_model.dense_units] * world_model_config.reward_model.mlp_layers,
            activation=nn.SiLU,
            layer_args={"bias": True},
            flatten_dim=None,
            norm_layer=norm_cls,
            norm_args={"normalized_shape": world_model_config.discount_model.dense_units}
        ).to(config.device)

        continue_model = MLP(
            input_dims=latent_state_size,
            output_dim=1,
            hidden_sizes=[world_model_config.discount_model.dense_units] * world_model_config.discount_model.mlp_layers,
            activation=nn.SiLU,
            layer_args={"bias": True},
            flatten_dim=None,
            norm_layer=norm_cls,
            norm_args={"normalized_shape": world_model_config.discount_model.dense_units},
        ).to(config.device)
        world_model = WorldModel(
            encoder.apply(init_weights),
            rssm,
            observation_model.apply(init_weights),
            reward_model.apply(init_weights),
            continue_model.apply(init_weights),
        )

        actor_cls = Actor
        actor: Actor = actor_cls(
            latent_state_size=latent_state_size,
            actions_dim=actions_dim,
            is_continuous=is_continuous,
            init_std=actor_config.init_std,
            min_std=actor_config.min_std,
            dense_units=actor_config.dense_units,
            activation=nn.SiLU,
            mlp_layers=actor_config.mlp_layers,
            distribution_config=config.distribution,
            norm_cls=norm_cls,
            unimix=config.unimix,
            action_clip=actor_config.action_clip,
        ).to(config.device)

        critic = MLP(
            input_dims=latent_state_size,
            output_dim=critic_config.bins,
            hidden_sizes=[critic_config.dense_units] * critic_config.mlp_layers,
            activation=nn.SiLU,
            layer_args={"bias": True},
            flatten_dim=None,
            norm_layer=norm_cls,
            norm_args={
                "normalized_shape": critic_config.dense_units,
            },
        ).to(config.device)

        if config.trunc_normal_init:
            world_model.apply(trunc_normal_init_weights())
            actor.apply(trunc_normal_init_weights())
            critic.apply(trunc_normal_init_weights())

        player = PlayerDV3(  # encoder, rssm, actor
            copy.deepcopy(world_model.encoder),
            copy.deepcopy(world_model.rssm),
            copy.deepcopy(actor),
            actions_dim,
            config.parallels,
            config.world_model.stochastic_size,
            config.world_model.recurrent_model.recurrent_state_size,
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
