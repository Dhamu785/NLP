# %% import libs
import torch as t
# %%
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
t.set_default_device(DEVICE)

print(t.get_default_device())
# %%
import string
import unicodedata

allowed_char = string.ascii_letters + ".,;'" + "_"
print(allowed_char)
# %% unicode to Ascii
def unicode_to_ascii(txt: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn' and c in allowed_char)

test = 'Ślusàrski'
out = unicode_to_ascii(test)
print(out)
# %%

def encode_text(txt: str) -> t.Tensor:
    