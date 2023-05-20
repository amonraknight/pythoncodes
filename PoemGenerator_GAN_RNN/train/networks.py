import torch.nn as nn
import torch
import random


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


class RNN2(nn.Module):
    def __init__(self, char_size, embedding_size, hidden_size, output_size, dropout=0.1,
                 layer_size=1, embedding_tensor=None):
        super(RNN2, self).__init__()
        self.hidden_size = hidden_size

        if embedding_tensor is None:
            self.embedding = nn.Embedding(char_size, embedding_size)
        else:
            self.embedding = nn.Embedding(num_embeddings=embedding_tensor.shape[0],
                                          embedding_dim=embedding_tensor.shape[1])
            self.embedding.weight = torch.nn.Parameter(embedding_tensor)

        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=layer_size, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input_tensor):
        embedded = self.dropout(self.embedding(input_tensor))
        output, _ = self.rnn(embedded)
        # What is the output shape? batch * sequence * hidden_size
        output = output[:, -1, ...]
        output = self.fc(output)
        output = self.softmax(output)
        return output


class RNN3(nn.Module):
    def __init__(self, char_size, embedding_size, hidden_size, output_size, dropout=0.1,
                 layer_size=1, embedding_tensor=None, leak=0.2):
        super(RNN3, self).__init__()
        self.hidden_size = hidden_size

        if embedding_tensor is None:
            self.embedding = nn.Embedding(char_size, embedding_size)
        else:
            self.embedding = nn.Embedding(num_embeddings=embedding_tensor.shape[0],
                                          embedding_dim=embedding_tensor.shape[1])
            self.embedding.weight = torch.nn.Parameter(embedding_tensor)

        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=layer_size, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.batchNorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.LeakyReLU(leak)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        embedded = self.dropout(self.embedding(input_tensor))
        output, _ = self.rnn(embedded)
        # What is the output shape? batch * sequence * hidden_size
        output = output[:, -1, ...]
        output = self.batchNorm(output)
        output = self.relu(output)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


class Encoder1(nn.Module):
    def __init__(self, embedding, hidden_size, layer_size=1, dropout=0.1):
        super(Encoder1, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding.embedding_dim, hidden_size, num_layers=layer_size, bidirectional=False,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        embedded = self.dropout(self.embedding(input_tensor))
        _, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder1(nn.Module):
    def __init__(self, output_size, embedding, hidden_size, layer_size=1, dropout=0.1):
        super(Decoder1, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding.embedding_dim, hidden_size, num_layers=layer_size, bidirectional=False,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    # The hidden and cell are from the encoder.
    def forward(self, input_tensor, hidden, cell):
        # input_tensor (batch, 1)
        embedded = self.dropout(self.embedding(input_tensor))
        # embedded (batch, 1, embed size 300)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        output = output[:, -1, ...]
        prediction = self.fc(output)
        return prediction, hidden, cell


class Seq2SeqNet1(nn.Module):
    def __init__(self, char_size, encode_dim, hidden_size, dropout, device,
                 encoder_layer_size=1, decoder_layer_size=1, embedding_tensor=None):
        super(Seq2SeqNet1, self).__init__()  # If not having self here, not able to correctly set the embedding

        if embedding_tensor is None:
            self.embedding = nn.Embedding(char_size, encode_dim)
        else:
            self.embedding = nn.Embedding(num_embeddings=embedding_tensor.shape[0],
                                          embedding_dim=embedding_tensor.shape[1])
            self.embedding.weight = torch.nn.Parameter(embedding_tensor)

        self.encoder = Encoder1(self.embedding, hidden_size, encoder_layer_size, dropout).to(device=device)
        self.decoder = Decoder1(char_size, self.embedding, hidden_size, decoder_layer_size, dropout).to(device=device)
        self.device = device
        self.output_char_size = char_size

    def forward(self, input_tensor, target=None, teacher_forcing_ratio=0.5, max_gen_length=15):
        batch_size = input_tensor.shape[0]
        if target is not None:
            # On training:
            target_length = target.shape[1]
            # To be transposed to (batch_size, target_length, self.output_char_size)
            outputs = torch.zeros(target_length, batch_size, self.output_char_size).to(self.device)
            hidden, cell = self.encoder(input_tensor)

            # The first char is the <start>
            input_t = target[:, 0].unsqueeze(1)
            for t in range(target_length):
                output, hidden, cell = self.decoder(input_t, hidden, cell)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio

                most_likely_idx = output.argmax(1).unsqueeze(1)
                input_t = target[:, t].unsqueeze(1) if teacher_force else most_likely_idx

            outputs = outputs.transpose(0, 1)
        else:
            # On generation.
            outputs = torch.zeros(max_gen_length, batch_size, self.output_char_size).to(self.device)
            hidden, cell = self.encoder(input_tensor)
            # Just to get a column of <start>.
            input_t = input_tensor[:, 0].unsqueeze(1)
            for i in range(max_gen_length):
                output, hidden, cell = self.decoder(input_t, hidden, cell)
                outputs[i] = output
                most_likely_idx = output.argmax(1)

                input_t = most_likely_idx.unsqueeze(1)

            outputs = outputs.transpose(0, 1)

        return outputs


class Decoder2(nn.Module):
    def __init__(self, output_size, embedding, hidden_size, layer_size=1, dropout=0.1, leak=0.2):
        super(Decoder2, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding.embedding_dim, hidden_size, num_layers=layer_size, bidirectional=False,
                           batch_first=True)
        self.batchNorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.LeakyReLU(leak)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    # The hidden and cell are from the encoder.
    def forward(self, input_tensor, hidden, cell):
        # input_tensor (batch, 1)
        embedded = self.dropout(self.embedding(input_tensor))
        # embedded (batch, 1, embed size 300)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        output = output[:, -1, ...]
        output = self.batchNorm(output)
        output = self.relu(output)
        prediction = self.fc(output)
        return prediction, hidden, cell


class Seq2SeqNet2(nn.Module):
    def __init__(self, char_size, encode_dim, hidden_size, dropout, device,
                 encoder_layer_size=1, decoder_layer_size=1, embedding_tensor=None, leak=0.2):
        super(Seq2SeqNet2, self).__init__()  # If not having self here, not able to correctly set the embedding

        if embedding_tensor is None:
            self.embedding = nn.Embedding(char_size, encode_dim)
        else:
            self.embedding = nn.Embedding(num_embeddings=embedding_tensor.shape[0],
                                          embedding_dim=embedding_tensor.shape[1])
            self.embedding.weight = torch.nn.Parameter(embedding_tensor)

        self.encoder = Encoder1(self.embedding, hidden_size, encoder_layer_size, dropout)
        self.decoder = Decoder2(char_size, self.embedding, hidden_size, decoder_layer_size, dropout, leak)
        self.output_char_size = char_size
        self.device = device

    def forward(self, input_tensor, target=None, teacher_forcing_ratio=0.5, max_gen_length=15):
        batch_size = input_tensor.shape[0]
        if target is not None:
            # On training:
            target_length = target.shape[1]
            # To be transposed to (batch_size, target_length, self.output_char_size)
            outputs = torch.zeros(target_length, batch_size, self.output_char_size, device=self.device)
            hidden, cell = self.encoder(input_tensor)

            # The first char is the <start>
            input_t = target[:, 0].unsqueeze(1)
            for t in range(target_length):
                output, hidden, cell = self.decoder(input_t, hidden, cell)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                most_likely_idx = output.argmax(1).unsqueeze(1)
                input_t = target[:, t].unsqueeze(1) if teacher_force else most_likely_idx

            outputs = outputs.transpose(0, 1)
        else:
            # On generation.
            outputs = torch.zeros(max_gen_length, batch_size, self.output_char_size, device=self.device)
            hidden, cell = self.encoder(input_tensor)
            # Just to get a column of <start>.
            input_t = input_tensor[:, 0].unsqueeze(1)
            for i in range(max_gen_length):
                output, hidden, cell = self.decoder(input_t, hidden, cell)
                outputs[i] = output
                most_likely_idx = output.argmax(1)

                input_t = most_likely_idx.unsqueeze(1)

            outputs = outputs.transpose(0, 1)

        return outputs
