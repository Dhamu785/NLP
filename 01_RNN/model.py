import torch as t 
import torch.nn as nn
import torch.nn.functional as F

class rnn_model(nn.Module):
    def __init__(self, input_shape: int, hidden_shape: int, output_shape: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=input_shape, hidden_size=hidden_shape)
        self.linear = nn.Linear(in_features=hidden_shape, out_features=output_shape)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        outs, hidden = self.rnn(x)
        output_logits = self.linear(hidden[0])
        output = self.softmax(output_logits)
        return output

if __name__ == "__main__":
    x = t.randn((5,1,10))
    model = rnn_model(10, 20, 5)
    output = model(x)
    print(output.shape)