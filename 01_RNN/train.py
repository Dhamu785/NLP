from trainer import train
from model import rnn_model
from get_data import helper, NameDataset

# data = NameDataset('/Users/dhamodharan/My-Python/AI-Tutorials/04_NLP/NLP/01_RNN/data/names')
data = NameDataset('C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\04\\NLP\\01_RNN\\data')
input_len = helper().len_allowed_char
hidden = 128
out_len = len(data.unique_labels)
print(f"{len(data) = }, {out_len = }, {input_len = }")

model = rnn_model(input_len, hidden, out_len)
# print(model)

all_losses = train(model, data)
print(all_losses)