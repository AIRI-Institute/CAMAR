import jax

ENV_DEVICE = jax.devices(jax.default_backend())[0]

if jax.default_backend() != "cpu":
    PREGEN_DEVICE = jax.devices("cpu")[0]
else:
    PREGEN_DEVICE = ENV_DEVICE
