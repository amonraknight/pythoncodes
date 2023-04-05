import torch.nn as nn


class RNN1(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, dropout=0.1):
        super(RNN1, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input_tensor):
        output, _ = self.rnn(input_tensor)
        output = self.dropout(output)
        # What is the output shape? batch * sequence * hidden_size
        output = output[:, -1, ...]
        output = self.fc(output)
        output = self.softmax(output)
        return output
