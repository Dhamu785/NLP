# %% import libs
import torch as t
import string
import unicodedata

# %% Device allocation
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
t.set_default_device(DEVICE)
print(t.get_default_device())

# %% Allowed characters and unicode to Ascii
allowed_char = string.ascii_letters + ".,;'" + "_"
len_allowed_char = len(allowed_char)
print(f"{'Allowed chars':<20} : {allowed_char}")
print(f"{'Lennght':<20} : {len(allowed_char)}")

# %%
def unicode_to_ascii(txt: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn' and c in allowed_char)

def encode_text(txt: str) -> t.Tensor:
    asci = unicode_to_ascii(txt)
    print(f"{'Befor asci':<20} : {txt}")
    print(f"{'After asci':<20} : {asci}")
    tensor = t.zeros(len(asci), 1, len_allowed_char)
    for idx, l in enumerate(asci):
        if l in allowed_char:
            l_idx = allowed_char.find(l)
        else:
            l_idx = allowed_char.find('_')
        tensor[idx][0][l_idx] = 1
    return tensor

test_w = 'Ślusàrski'
t_out = encode_text(test_w)
print(t_out)
# %%
