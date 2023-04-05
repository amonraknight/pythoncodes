import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as datautil
from torch import optim

import traindata as tdata
from networks import RNN1
import common.config as config
from common.module_saveload import ModuleSaveLoad


def train():

    # Set device
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")

    # Module save-loader
    saveLoader = ModuleSaveLoad()

    # Set up the model, loss and optimizer
    rnn = RNN1(config.EMBEDDING_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE, config.DROP_OUT)
    rnn.to(device=device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=config.LEARNING_RATE)

    # Load train/test data into data loaders
    X_train, Y_train, X_validate, Y_validate = tdata.prepare_train_test_data()
    X_train = torch.FloatTensor(X_train).to(device=device)
    Y_train = torch.LongTensor(Y_train).to(device=device)
    X_validate = torch.FloatTensor(X_validate).to(device=device)
    Y_validate = torch.LongTensor(Y_validate).to(device=device)
    train_ds = datautil.TensorDataset(X_train, Y_train)
    # num_workers must be 0 on Windows, otherwise the data will get lost when enumerate
    train_loader = datautil.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    # num_workers must be 0 on Windows, otherwise the data will get lost when enumerate
    validate_ds = datautil.TensorDataset(X_validate, Y_validate)
    validate_loader = datautil.DataLoader(validate_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    # Loop epoch
    for e in range(config.EPOCH):
        train_loss = []
        #Why values are lost?
        for batch_idx, (data, label) in enumerate(train_loader):
            rnn.train()

            # data's shape is batch_size * sequence_size * embedding_size
            output = rnn(data)
            loss = criterion(output, label)
            train_loss.append(loss.detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        valid_loss = []
        rights = 0
        for batch_idx, (data, label) in enumerate(validate_loader):
            rnn.eval()
            output = rnn(data)
            loss = criterion(output, label)
            valid_loss.append(loss.detach().cpu().numpy())
            max_idx = torch.max(output, 1).indices
            difference = np.subtract(max_idx.detach().cpu().numpy(), label.detach().cpu().numpy())
            difference = np.absolute(difference)
            correct = np.sum(difference)
            correct = label.shape[0] - correct
            rights += correct

        print("Epoch {}, train loss {:.2f}, validate loss {:.2f}, validate accuracy {:.2f}"
              .format(e, np.mean(train_loss), np.mean(valid_loss), rights/validate_ds.__len__()))

        saveLoader.save_module(rnn, e)


if __name__ == "__main__":
    train()



