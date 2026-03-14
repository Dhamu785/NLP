# %% import libs
import torch as t
# %%
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
t.set_default_device(DEVICE)

print(t.get_default_device())
# %%
