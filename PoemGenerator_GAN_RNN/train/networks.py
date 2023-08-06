import torch.nn as nn
import torch
import random
import math
import torch.nn.functional as F

from torch import Tensor


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


# taking in the char index
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


# take in the embedded
class RNN4(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, dropout=0.1, layer_size=1, leak=0.2):
        super(RNN4, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=layer_size, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.batchNorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.LeakyReLU(leak)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedded_input):
        embedded = self.dropout(embedded_input)
        output, _ = self.rnn(embedded)
        # What is the output shape? batch * sequence * hidden_size
        output = output[:, -1, ...]
        output = self.batchNorm(output)
        output = self.relu(output)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


# take in the one-hot in (char_size)
class RNN5(nn.Module):
    def __init__(self, char_size, hidden_size, output_size, dropout=0.1, layer_size=1, leak=0.2):
        super(RNN5, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(char_size, hidden_size, num_layers=layer_size, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.batchNorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.LeakyReLU(leak)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        output, _ = self.rnn(input_tensor)
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


# The inbound should be embedded.
class Decoder3(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_size, layer_size=1):
        super(Decoder3, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        # self.embedding = embedding
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=layer_size, bidirectional=False,
                           batch_first=True)
        self.batchNorm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    # The hidden and cell are from the encoder.
    def forward(self, input_tensor, hidden, cell):
        # input_tensor (batch, 1)
        # embedded = self.dropout(self.embedding(input_tensor))
        # input_tensor should be in (batch, 1, embed size 300)
        input_tensor = input_tensor.unsqueeze(1)

        output, (hidden, cell) = self.rnn(input_tensor, (hidden, cell))
        output = output[:, -1, ...]
        output = self.batchNorm(output)
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


# input_tensor is the indexes in (batch_size, sequence_length)
# outputs is in (batch_size, sequence_length, char_size)
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
            input_t = target[:, 0]
            outputs[0] = F.one_hot(input_t, num_classes=self.output_char_size)
            for t in range(1, target_length):
                output, hidden, cell = self.decoder(input_t.unsqueeze(1), hidden, cell)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                most_likely_idx = output.argmax(1)
                input_t = target[:, t] if teacher_force else most_likely_idx

            outputs = outputs.transpose(0, 1)
        else:
            # On generation.
            outputs = torch.zeros(max_gen_length, batch_size, self.output_char_size, device=self.device)
            hidden, cell = self.encoder(input_tensor)
            # Just to get a column of <start>.
            input_t = input_tensor[:, 0]
            outputs[0] = F.one_hot(input_t, num_classes=self.output_char_size)
            for i in range(1, max_gen_length):
                output, hidden, cell = self.decoder(input_t.unsqueeze(1), hidden, cell)
                outputs[i] = output
                most_likely_idx = output.argmax(1)

                input_t = most_likely_idx

            outputs = outputs.transpose(0, 1)

        return outputs.contiguous()


# The output of this network should be the embedding vector instead of the one-hot.
class Seq2SeqNet3(nn.Module):
    def __init__(self, char_size, encode_dim, hidden_size, dropout, device,
                 encoder_layer_size=1, decoder_layer_size=1, embedding_tensor=None):
        super(Seq2SeqNet3, self).__init__()  # If not having self here, not able to correctly set the embedding

        if embedding_tensor is None:
            self.embedding = nn.Embedding(char_size, encode_dim)
        else:
            self.embedding = nn.Embedding(num_embeddings=embedding_tensor.shape[0],
                                          embedding_dim=embedding_tensor.shape[1])
            self.embedding.weight = torch.nn.Parameter(embedding_tensor)

        self.encoder = Encoder1(self.embedding, hidden_size, encoder_layer_size, dropout)
        self.decoder = Decoder3(encode_dim, encode_dim, hidden_size, decoder_layer_size)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.embedding_size = encode_dim

    def forward(self, input_tensor, target=None, teacher_forcing_ratio=0.5, max_gen_length=9):
        batch_size = input_tensor.shape[0]
        hidden, cell = self.encoder(input_tensor)

        if target is not None:
            # On training:
            target_length = target.shape[1]
            # To be transposed to (batch_size, target_length, self.output_char_size)
            outputs = torch.zeros(target_length, batch_size, self.embedding_size, device=self.device)

            # Just to get a column of <start>. The input to the decoder should be the embedded vector.
            # (batch_size, embedding_size)
            input_t = torch.zeros(batch_size, dtype=torch.int64, device=self.device) + 2
            input_embedded = self.embedding(input_t)
            # Set the initial character to <start>.
            outputs[0] = input_embedded
            for t in range(1, target_length):
                # Embed input
                # Why sometimes having
                output, hidden, cell = self.decoder(input_embedded, hidden, cell)
                # Ensure the outputs are within (-1,1)
                output = self.sigmoid(output) * 2 - 1
                outputs[t] = output
                if random.random() < teacher_forcing_ratio:
                    input_embedded = self.embedding(target[:, t])
                else:
                    input_embedded = output

            outputs = outputs.transpose(0, 1)
        else:
            # On generation.
            outputs = torch.zeros(max_gen_length, batch_size, self.embedding_size, device=self.device)

            # Just to get a column of <start>. The input to the decoder should be the embedded vector.
            # (batch_size, embedding_size)
            input_t = torch.zeros(batch_size, dtype=torch.int64, device=self.device) + 2
            input_embedded = self.embedding(input_t)
            outputs[0] = input_embedded
            for i in range(1, max_gen_length):
                output, hidden, cell = self.decoder(input_embedded, hidden, cell)
                # Ensure the outputs are within (-1,1)
                output = self.sigmoid(output) * 2 - 1
                outputs[i] = output
                input_embedded = output

            outputs = outputs.transpose(0, 1)

        return outputs


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


# SeqGAN networks:
# Skip the discriminator and reuse RNN3.
# The generator for SeqGAN:
class Seq2SeqNet4(nn.Module):
    def __init__(self, char_size, embed_dim, hidden_size, dropout, device,
                 encoder_layer_size=1, decoder_layer_size=1, embedding_tensor=None, leak=0.2):
        super(Seq2SeqNet4, self).__init__()  # If not having self here, not able to correctly set the embedding

        if embedding_tensor is None:
            self.embedding = nn.Embedding(char_size, embed_dim)
        else:
            self.embedding = nn.Embedding(num_embeddings=embedding_tensor.shape[0],
                                          embedding_dim=embedding_tensor.shape[1])
            self.embedding.weight = torch.nn.Parameter(embedding_tensor)

        self.encoder = Encoder1(self.embedding, hidden_size, encoder_layer_size, dropout)
        self.decoder = Decoder2(char_size, self.embedding, hidden_size, decoder_layer_size, dropout, leak)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.output_char_size = char_size
        self.device = device

    # x is 2-D (batch_size, 1)
    # hidden and cell are 3-D
    def step(self, x, hidden, cell):
        output, hidden, cell = self.decoder(x, hidden, cell)
        output = self.softmax(output)
        return output, hidden, cell

    def forward(self, input_tensor, target=None, teacher_forcing_ratio=0.5, max_gen_length=9):
        batch_size = input_tensor.shape[0]
        hidden, cell = self.encoder(input_tensor)

        if target is not None:
            # On training:
            target_length = target.shape[1]
            # To be transposed to (batch_size, target_length, self.output_char_size)
            outputs = torch.zeros(target_length, batch_size, self.output_char_size, device=self.device)

            # input_t is the first char is the <start>
            # input_t in shape (batch_size) and having the indexes
            input_t = target[:, 0]
            outputs[0] = F.one_hot(input_t, num_classes=self.output_char_size)
            for t in range(1, target_length):
                output, hidden, cell = self.step(input_t.unsqueeze(1), hidden, cell)
                outputs[t] = output
                most_likely_idx = output.argmax(1)
                teacher_force = random.random() < teacher_forcing_ratio
                input_t = target[:, t] if teacher_force else most_likely_idx

        else:
            # On generation.
            outputs = torch.zeros(max_gen_length, batch_size, self.output_char_size, device=self.device)
            # Just to get a column of <start>.
            input_t = input_tensor[:, 0]
            outputs[0] = F.one_hot(input_t, num_classes=self.output_char_size)
            for i in range(1, max_gen_length):
                output, hidden, cell = self.step(input_t.unsqueeze(1), hidden, cell)
                outputs[i] = output
                most_likely_idx = output.argmax(1)

                input_t = most_likely_idx

        # outputs is in (seq_len, batch, char_size)
        outputs = outputs.transpose(0, 1).contiguous()
        return outputs

    # input_tensor is the upper context.
    # x is the indexes of a given part of the text
    def sample(self, input_tensor, batch_size, seq_len, x=None):
        hidden, cell = self.encoder(input_tensor)
        no_input_flag = x is None
        # outputs is keeping the indexes.
        outputs = torch.zeros(seq_len, batch_size, dtype=torch.int64, device=self.device)
        if no_input_flag:
            x = input_tensor[:, 0]  # get <start>
            outputs[0] = x
            for i in range(1, seq_len):
                x = x.unsqueeze(1)
                x, hidden, cell = self.step(x, hidden, cell)
                x = x.argmax(1)
                outputs[i] = x
        else:
            given_len = x.shape[1]
            for i in range(0, given_len):
                given_input = x[:, i]
                given_input = given_input.unsqueeze(1)
                output, hidden, cell = self.step(given_input, hidden, cell)
                outputs[i] = x[:, i]
                output = output.argmax(1)

            for i in range(given_len, seq_len):
                outputs[i] = output
                given_input = output
                given_input = given_input.unsqueeze(1)
                if i < seq_len - 1:
                    output, hidden, cell = self.step(given_input, hidden, cell)
                    output = output.argmax(1)

        return outputs.transpose(0, 1).contiguous()


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""

    def __init__(self, device):
        super(GANLoss, self).__init__()
        self.device = device

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        char_size = prob.size(1)

        one_hot = F.one_hot(target, num_classes=char_size)
        one_hot = one_hot.to(self.device)
        loss = prob * one_hot
        loss = torch.matmul(loss.transpose(0, 1), reward)
        loss = -torch.sum(loss)
        return loss.contiguous()


'''
    Transformer
    Build a model having encoder, decoder, positional encoding and embedding.
'''


# The learned positional encoding usually has a better result than the static one.
class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 10000^(2i/dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(-2)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # embedding 是分布是N(0,1)，乘上math.sqrt(self.emb_size)，增加var
        # 增加差异？
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# https://zhuanlan.zhihu.com/p/430893933
class Transformer3(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, src_vocab_size,
                 tgt_vocab_size, dim_feedforward: int = 512, dropout: float = 0.2):
        super(Transformer3, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_token_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_token_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_token_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_token_emb(tgt))

        # memory_mask设置为None,不需要mask; memory=encoder(input)
        # memory_key_padding_mask 和 src_padding_mask 一样，
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_token_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_token_emb(tgt)), memory, tgt_mask)

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
