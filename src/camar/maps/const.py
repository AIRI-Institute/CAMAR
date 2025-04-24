import jax

CPU_DEVICE = jax.devices("cpu")[0]
GPU_DEVICE = jax.devices("cuda")[0]
