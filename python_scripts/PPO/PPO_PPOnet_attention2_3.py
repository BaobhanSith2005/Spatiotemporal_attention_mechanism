from python_scripts.PPO.PPO_PPOnet_attention4_1 import PPO
from python_scripts.PPO.PPO_PPOnet_attention_original_base import StrictFourOneVariantActorCriticBase


class ActorCritic(StrictFourOneVariantActorCriticBase):
    image_spatial_mode = "conv"
    state_spatial_mode = "gnn"
    image_temporal_mode = "attention"
    state_temporal_mode = "attention"
