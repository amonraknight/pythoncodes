import torch.nn as nn
import torch
import random
import math
import torch.nn.functional as F


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

    def forward(self, input_tensor, target=None, teacher_forcing_ratio=0.5, max_gen_length=9):
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


# PositionalEncoding add the position info to the inbound.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
        Container module with an encoder, a recurrent or transformer module, and a decoder.
        Original parameters: ntoken, ninp, nhead, nhid, nlayers, dropout=0.5
    """
    def __init__(self, char_size, encode_dim, head_num, hidden_size, layer_num, dropout=0.5, embedding_tensor=None):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(encode_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(encode_dim, head_num, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layer_num)

        if embedding_tensor is None:
            self.encoder = nn.Embedding(char_size, encode_dim)
        else:
            self.encoder = nn.Embedding(num_embeddings=embedding_tensor.shape[0],
                                          embedding_dim=embedding_tensor.shape[1])
            self.encoder.weight = torch.nn.Parameter(embedding_tensor)

        self.encode_dim = encode_dim
        self.decoder = nn.Linear(encode_dim, char_size)
        self.init_weights(embedding_tensor is None)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self, init_encoder=False):
        initrange = 0.1
        if init_encoder:
            nn.init.uniform_(self.encoder.weight, -initrange, initrange)

        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.encode_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        # Return size is (batch_size, sequence_length ,char_size) / (8, 9, NNNNNN).
        return F.log_softmax(output, dim=-1)

