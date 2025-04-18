import jax
import jax.numpy as jnp

import sys
print("SYS AR")

print("JAX version:", jax.__version__)
print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)

# Test d'opération simple
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.sin(x)
print("sin(x):", y)
