import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
import optax

from reward import RewardModel  # NOTE: change to a simple reward function (non-transformer-based)

BETA = 0.01
MAX_NEW_TOKENS = 8  # placeholder
MU = 4
G = 8

def dapo_loss(log_probs_rl , old_log_probs , advantages , epsilon_low=0.2 , epsilon_high=0.28):
    ratio = jnp.exp(log_probs_rl - old_log_probs)  # [batch , G , response_len]
    At = rearrange(advantages , 'b g -> b g 1')

    clipped = jnp.clip(ratio , 1-epsilon_low , 1+epsilon_high)
    surrogate = jnp.minimum(ratio * At , clipped * At)

    loss = -surrogate.mean()
    return loss

@nnx.jit(static_argnames=('prompt_len',))
def train_step(policy , optimizer , outputs , old_log_probs , log_probs_sft , advantages , prompt_len):
    def loss_fn(policy):
        log_probs_rl = compute_log_probs(policy , outputs , prompt_len)
        return dapo_loss(log_probs_rl , old_log_probs , log_probs_sft , advantages)

    loss_val , grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(policy , grads)
    return loss_val
