import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as datautil
from torch import optim
from tqdm import tqdm
import itertools

import traindata as tdata
from networks import RNN3, RNN5, Seq2SeqNet2, Seq2SeqNet4, GANLoss
import common.config as config
from common.module_saveload import ModuleSaveLoad
from utilities.evaluatior_util import valuate_generator, generate_random_poems, watch_grad, require_grad
from rollout import Rollout

'''
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
        # Why values are lost?
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
              .format(e, np.mean(train_loss), np.mean(valid_loss), rights / validate_ds.__len__()))

        saveLoader.save_module(rnn, e)


# Train a line to line as a part of poem generator.
def train_2(manual_test=False):
    # Set device
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")

    # Prepare train data
    context_target_index_list, char_array, vectors = tdata.prepare_train_test_data_2()
    embedding_tensor = torch.from_numpy(vectors).to(device=device)
    # Take 100 for debug
    # context_target_index_list = context_target_index_list[0:100]

    # Module save-loader
    save_loader = ModuleSaveLoad()
    min_loss = 1000

    # Initialize model:
    model = Seq2SeqNet1(char_array.size, config.EMBEDDING_SIZE, config.HIDDEN_SIZE_2, config.DROP_OUT, device,
                        encoder_layer_size=config.LAYER_SIZE_2, decoder_layer_size=config.LAYER_SIZE_2,
                        embedding_tensor=embedding_tensor)

    saved_state_dict, starting_episode = save_loader.load_module()
    if saved_state_dict is not None and starting_episode is not None:
        model.load_state_dict(saved_state_dict)
    else:
        starting_episode = -1

    if manual_test:
        model.eval()
        manual_test_lines = []
        for each_line in config.MANUAL_TEST_LINES:
            manual_test_lines.append(tdata.convert_line_to_indexes(each_line, char_array))

        manual_test_lines = torch.LongTensor(np.array(manual_test_lines)).to(device=device)
        valuate_generator(model, manual_test_lines, char_array, None)
    else:
        # Save 5 for validation
        val_idx, train_idx = datautil.random_split(context_target_index_list,
                                                   [5, context_target_index_list.shape[0] - 5])
        train_x = torch.LongTensor(context_target_index_list[train_idx.indices][:, 0]).to(device=device)
        train_y = torch.LongTensor(context_target_index_list[train_idx.indices][:, 1]).to(device=device)

        val_x = torch.LongTensor(context_target_index_list[val_idx.indices][:, 0]).to(device=device)
        val_y = torch.LongTensor(context_target_index_list[val_idx.indices][:, 1]).to(device=device)

        train_ds = datautil.TensorDataset(train_x, train_y)
        # num_workers must be 0 on Windows, otherwise the data will get lost when enumerate
        train_loader = datautil.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

        # Criterion
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        # criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()

        # train loop
        for e in range(starting_episode + 1, config.EPOCH):
            model.train()
            total_loss = 0
            for i, (src, tar) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                output = model(src, tar, teacher_forcing_ratio=config.TEACHER_FORCING_RATE)

                # output    (batch, sequence, char_size)    --> (batch * sequence, char size)
                # tar       (batch, sequence, 1)            --> (batch * sequence, 1)
                output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
                tar = tar.reshape(tar.shape[0] * tar.shape[1])
                loss = criterion(output, tar)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
                loss.backward()
                optimizer.step()
                total_loss += loss

            total_loss = total_loss / i
            print('Epoch {}, train total loss {:.5f}'.format(e, total_loss))
            if min_loss > total_loss:
                save_loader.save_module(model, e)
                min_loss = total_loss

            # test
            valuate_generator(model, val_x, char_array, val_y)





# Train a random poem generator with GAN
def train_3():
    # Set device
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")

    # Prepare train data
    context_target_index_list, char_array, vectors = tdata.prepare_train_test_data_2()
    embedding_tensor = torch.from_numpy(vectors).to(device=device)
    # Take 100 for debug
    # context_target_index_list = context_target_index_list[0:100]

    # discriminator model
    discriminator_model = RNN2(char_array.size, config.EMBEDDING_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE,
                               config.DROP_OUT, layer_size=config.LAYER_SIZE_D, embedding_tensor=embedding_tensor)
    discriminator_model.to(device=device)

    # generator model
    generator_model = Seq2SeqNet1(char_array.size, config.EMBEDDING_SIZE, config.HIDDEN_SIZE_2, config.DROP_OUT, device,
                                  encoder_layer_size=config.LAYER_SIZE_G, decoder_layer_size=config.LAYER_SIZE_G,
                                  embedding_tensor=embedding_tensor)

    # loss and optimizer
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer_d = optim.Adam(discriminator_model.parameters(), lr=config.LEARNING_RATE)
    optimizer_g = optim.Adam(generator_model.parameters(), lr=config.LEARNING_RATE)

    g_train_idx, train_idx = datautil.random_split(context_target_index_list,
                                                   [1000, context_target_index_list.shape[0] - 1000],
                                                   generator=torch.Generator().manual_seed(42))
    train_x = torch.LongTensor(context_target_index_list[train_idx.indices, 0]).to(device=device)
    train_y = torch.LongTensor(context_target_index_list[train_idx.indices, 1]).to(device=device)
    train_ds = datautil.TensorDataset(train_x, train_y)
    # num_workers must be 0 on Windows, otherwise the data will get lost when enumerate
    train_loader = datautil.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    # Prepare a group of poem lines never used in generator learning. Use itertools.cycle to guarantee the acquisition.
    g_train_x = torch.LongTensor(context_target_index_list[g_train_idx.indices, 0]).to(device=device)
    g_train_y = torch.LongTensor(context_target_index_list[g_train_idx.indices, 1]).to(device=device)
    g_train_ds = datautil.TensorDataset(g_train_x, g_train_y)
    g_train_loader = datautil.DataLoader(g_train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    g_train_dl_iter = itertools.cycle(g_train_loader)

    # save/load model
    save_loader_d = ModuleSaveLoad('_d')
    save_loader_g = ModuleSaveLoad('_g')

    saved_state_dict_d, _ = save_loader_d.load_module()
    saved_state_dict_g, starting_episode = save_loader_g.load_module()
    if saved_state_dict_d is not None and saved_state_dict_g is not None and starting_episode is not None:
        discriminator_model.load_state_dict(saved_state_dict_d)
        generator_model.load_state_dict(saved_state_dict_g)
    else:
        starting_episode = -1

    for e in range(starting_episode + 1, config.EPOCH):
        total_loss_d = 0
        total_loss_g = 0
        g_count = 0
        considered_real = 0

        for i, (real_poem1_idx, real_poem2_idx) in enumerate(tqdm(train_loader)):
            if real_poem2_idx.shape[0] != config.BATCH_SIZE:
                break

            generator_model.eval()
            # The output_1 is in (batch_size, , )
            g_output_1 = generator_model(real_poem1_idx, max_gen_length=9)
            # It should be of the same shape of real_poem_idx.
            g_output_1 = g_output_1.argmax(2)

            label_true = torch.ones(config.BATCH_SIZE, dtype=torch.int64, device=device)
            label_false = torch.zeros(config.BATCH_SIZE, dtype=torch.int64, device=device)

            d_input_1 = torch.cat((real_poem1_idx, g_output_1), dim=0)
            d_label = torch.cat((label_true, label_false), dim=0)

            # Train discriminator.
            discriminator_model.train()
            discriminator_model.zero_grad()
            output_2 = discriminator_model(d_input_1)
            loss_d = criterion(output_2, d_label)
            total_loss_d += loss_d
            loss_d.backward()
            optimizer_d.step()

            # Train g
            generator_model.train()
            if np.random.rand() < (e / config.EPOCH) * config.GAN_RATE_IDX:
                # Train as GAN
                g_count += 1
                g_x, g_y = next(g_train_dl_iter)
                g_output_2 = generator_model(g_x, max_gen_length=9)
                g_output_2 = g_output_2.argmax(2)

                # valuate_generator(generator_model, g_x, char_array, g_y)

                generator_model.zero_grad()
                d_output_2 = discriminator_model(g_output_2)
                # Generator wants them to be judged as real.
                loss_g = criterion(d_output_2, label_true)

                # Count correctness
                count_of_outputs_considered_as_real = d_output_2.argmax(1).sum()
                considered_real += count_of_outputs_considered_as_real

                total_loss_g += loss_g
                loss_g.backward()
                optimizer_g.step()
            else:
                # Tain with original poem
                generator_model.zero_grad()
                g_output_3 = generator_model(real_poem1_idx, real_poem2_idx,
                                             teacher_forcing_ratio=config.TEACHER_FORCING_RATE)
                # output    (batch, sequence, char_size)    --> (batch * sequence, char size)
                # tar       (batch, sequence, 1)            --> (batch * sequence, 1)
                g_output_3 = g_output_3.reshape(g_output_3.shape[0] * g_output_3.shape[1], g_output_3.shape[2])
                tar = real_poem2_idx.reshape(real_poem2_idx.shape[0] * real_poem2_idx.shape[1])
                loss_g = criterion(g_output_3, tar)
                torch.nn.utils.clip_grad_norm_(generator_model.parameters(), config.CLIP)
                loss_g.backward()
                optimizer_g.step()

        # Save the model in the end of each epoch.
        save_loader_d.save_module(discriminator_model, e)
        save_loader_g.save_module(generator_model, e)

        if g_count == 0:
            print('Epoch {}: avg_loss_d: {:.4f}'.format(e, total_loss_d / i))
        else:
            print('Epoch {}: avg_loss_d: {:.4f}, avg_loss_g: {:.4f}, considered_correct: {:.4f}'.format(e,
                                                                                                        total_loss_d / i,
                                                                                                        total_loss_g / g_count,
                                                                                                        considered_real / g_count / config.BATCH_SIZE))

        # Generate a sample.

        manual_test_lines = []
        for each_line in config.MANUAL_TEST_LINES:
            manual_test_lines.append(tdata.convert_line_to_indexes(each_line, char_array))
        manual_test_lines = torch.LongTensor(np.array(manual_test_lines)).to(device=device)
        valuate_generator(generator_model, manual_test_lines, char_array, None)

'''


def train_4(is_demo=False):
    # Set device
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")

    # Prepare train data
    context_target_index_list, char_array, vectors = tdata.prepare_train_test_data_2()
    char_size = char_array.size
    embedding_tensor = torch.from_numpy(vectors).to(device=device)
    # Take 100 for debug
    if config.IS_DEBUG:
        context_target_index_list = context_target_index_list[0:1000]

    # discriminator model
    discriminator_model = RNN5(char_size,
                               config.HIDDEN_SIZE,
                               config.OUTPUT_SIZE,
                               config.DROP_OUT,
                               layer_size=config.LAYER_SIZE_D,
                               leak=config.LEAK)
    discriminator_model.to(device=device)

    # generator model
    # Decoder-Encoder sequnece-2-sequence model:
    generator_model = Seq2SeqNet2(char_size,
                                  config.EMBEDDING_SIZE,
                                  config.HIDDEN_SIZE_2,
                                  config.DROP_OUT, device,
                                  encoder_layer_size=config.LAYER_SIZE_G,
                                  decoder_layer_size=config.LAYER_SIZE_G,
                                  embedding_tensor=embedding_tensor,
                                  leak=config.LEAK)
    generator_model.to(device=device)

    require_grad(generator_model)

    # loss and optimizer
    criterion_1 = nn.BCELoss()
    criterion_2 = nn.CrossEntropyLoss()
    optimizer_d = optim.Adam(discriminator_model.parameters(), lr=config.LEARNING_RATE_D)
    optimizer_g = optim.Adam(generator_model.parameters(), lr=config.LEARNING_RATE_G)

    train_x = torch.LongTensor(context_target_index_list[:, 0]).to(device=device)
    train_y = torch.LongTensor(context_target_index_list[:, 1]).to(device=device)
    train_ds = datautil.TensorDataset(train_x, train_y)
    # num_workers must be 0 on Windows, otherwise the data will get lost when enumerate
    train_loader = datautil.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    # save/load model
    save_loader_d = ModuleSaveLoad('_d')
    save_loader_g = ModuleSaveLoad('_g')

    saved_state_dict_d, _ = save_loader_d.load_module()
    saved_state_dict_g, starting_episode = save_loader_g.load_module()
    if not is_demo and saved_state_dict_d is not None:
        discriminator_model.load_state_dict(saved_state_dict_d)

    if saved_state_dict_g is not None and starting_episode is not None:
        generator_model.load_state_dict(saved_state_dict_g)
    else:
        starting_episode = -1

    if is_demo:
        manual_test_lines = []
        for each_line in config.MANUAL_TEST_LINES:
            manual_test_lines.append(tdata.convert_line_to_indexes(each_line, char_array))
        manual_test_lines = torch.LongTensor(np.array(manual_test_lines)).to(device=device)
        valuate_generator(generator_model, manual_test_lines, char_array, None)
    else:

        print('Starting epoch:{}'.format(starting_episode + 1))
        for e in range(starting_episode + 1, config.EPOCH):
            total_loss_d = 0
            total_loss_g = 0
            d_count = 0
            g_count = 0
            considered_real = 0

            for i, (real_poem1_idx, real_poem2_idx) in enumerate(tqdm(train_loader)):
                if real_poem2_idx.shape[0] != config.BATCH_SIZE:
                    break

                # real_poem?_idx is in (batch_size, sequence_length)
                '''
                random_input = generate_random_poems(batch_size=config.BATCH_SIZE,
                                                     char_size=char_array.size,
                                                     start_idx=2, end_idx=3, pad_idx=0,
                                                     dvc=device)
                '''

                # (batch_size, seq_len, char_size)
                g_output_1 = generator_model(real_poem1_idx)
                g_output_1 = F.gumbel_softmax(g_output_1.detach(), dim=-1)

                # transformer model
                # g_output_1 = generator_model(real_poem1_idx)
                label_true = torch.ones(config.BATCH_SIZE, dtype=torch.float32, device=device)

                # It should be of the same shape of real_poem_idx.

                label_false = torch.zeros(config.BATCH_SIZE, dtype=torch.float32, device=device)

                real_poem2_onehot = F.one_hot(real_poem2_idx, num_classes=char_size)
                d_input_1 = torch.cat((real_poem2_onehot, g_output_1), dim=0)
                d_label = torch.cat((label_true, label_false), dim=0)

                # Train discriminator.

                output_2 = discriminator_model(d_input_1).squeeze()
                loss_d = criterion_1(output_2, d_label)

                wrong_judge_value = (output_2 * torch.cat((label_false, label_true), dim=0)).mean() * 2
                # Catch up when not able to differentiate.
                if d_count == 0 or wrong_judge_value.item() > config.D_TRAIN_THRESHOLD:
                    d_count += 1

                    total_loss_d += loss_d
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator_model.parameters(), config.CLIP)
                    optimizer_d.step()

                # Train g
                if e > config.EPOCH * (1 - config.GAN_RATE_IDX):
                    # Train as GAN
                    g_count += 1

                    g_output_2 = generator_model(real_poem1_idx)

                    loss_g = 0
                    considered_real_e = 0
                    # Take gumbel max.
                    for _ in range(0, config.ROLLOUT_NUMBER):
                        gumbel_output = F.gumbel_softmax(g_output_2, dim=-1)
                        d_output_2 = discriminator_model(gumbel_output).squeeze()
                        loss_g += criterion_1(d_output_2, label_true)
                        considered_real_e += d_output_2.mean().item()

                    loss_g = loss_g / config.BATCH_SIZE / config.ROLLOUT_NUMBER
                    total_loss_g += loss_g

                    # Count correctness
                    considered_real += considered_real_e / config.ROLLOUT_NUMBER
                    optimizer_g.zero_grad()
                    loss_g.backward()
                    # watch_grad(generator_model)
                    torch.nn.utils.clip_grad_norm_(generator_model.parameters(), config.CLIP)
                    optimizer_g.step()
                else:
                    # Learn from the training data.
                    g_output_3 = generator_model(real_poem1_idx, real_poem2_idx,
                                                 teacher_forcing_ratio=config.TEACHER_FORCING_RATE)

                    g_output_3 = g_output_3.reshape(g_output_3.shape[0] * g_output_3.shape[1],
                                                    g_output_3.shape[2]).contiguous()
                    tar = real_poem2_idx.reshape(real_poem2_idx.shape[0] * real_poem2_idx.shape[1]).contiguous()

                    loss_g = criterion_2(g_output_3, tar)

                    optimizer_g.zero_grad()
                    loss_g.backward()
                    # watch_grad(generator_model)
                    torch.nn.utils.clip_grad_norm_(generator_model.parameters(), config.CLIP)
                    optimizer_g.step()

            # Save the model in the end of each epoch.
            save_loader_d.save_module(discriminator_model, e)
            save_loader_g.save_module(generator_model, e)

            if g_count == 0:
                print('Epoch {}: avg_loss_d: {:.4f}'.format(e, total_loss_d / d_count))
            else:
                print('Epoch {}: avg_loss_d: {:.4f}, avg_loss_g: {:.4f}, considered_correct: {:.4f}'.format(e,
                                                                                                            total_loss_d / d_count,
                                                                                                            total_loss_g / g_count,
                                                                                                            considered_real / g_count))

            # Generate a sample.
            # generator_model.eval()
            manual_test_lines = []
            for each_line in config.MANUAL_TEST_LINES:
                manual_test_lines.append(tdata.convert_line_to_indexes(each_line, char_array))
            manual_test_lines = torch.LongTensor(np.array(manual_test_lines)).to(device=device)
            valuate_generator(generator_model, manual_test_lines, char_array, None)


# SeqGAN
def train_5(is_demo=False):
    # Set device
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")

    # Prepare train data
    context_target_index_list, char_array, vectors = tdata.prepare_train_test_data_2()
    embedding_tensor = torch.from_numpy(vectors).to(device=device)

    # In test, minimize the batch.
    if config.IS_DEBUG:
        context_target_index_list = context_target_index_list[0:1000]

    # Discriminator
    discriminator_model = RNN3(char_array.size,
                               config.EMBEDDING_SIZE,
                               config.HIDDEN_SIZE,
                               config.OUTPUT_SIZE,
                               config.DROP_OUT,
                               layer_size=config.LAYER_SIZE_D,
                               embedding_tensor=embedding_tensor,
                               leak=config.LEAK)
    discriminator_model.to(device=device)

    # generator model
    # Decoder-Encoder sequnece-2-sequence model:
    generator_model = Seq2SeqNet4(char_array.size,
                                  config.EMBEDDING_SIZE,
                                  config.HIDDEN_SIZE_2,
                                  config.DROP_OUT, device,
                                  encoder_layer_size=config.LAYER_SIZE_G,
                                  decoder_layer_size=config.LAYER_SIZE_G,
                                  embedding_tensor=embedding_tensor,
                                  leak=config.LEAK)
    generator_model.to(device=device)

    # split train data to context and target.
    train_x = torch.LongTensor(context_target_index_list[:, 0]).to(device=device)
    train_y = torch.LongTensor(context_target_index_list[:, 1]).to(device=device)
    train_ds = datautil.TensorDataset(train_x, train_y)
    # num_workers must be 0 on Windows, otherwise the data will get lost when enumerate
    train_loader = datautil.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    # save/load model
    save_loader_d = ModuleSaveLoad('_d')
    save_loader_g = ModuleSaveLoad('_g')

    saved_state_dict_d, _ = save_loader_d.load_module()
    saved_state_dict_g, starting_episode = save_loader_g.load_module()
    if not is_demo and saved_state_dict_d is not None:
        discriminator_model.load_state_dict(saved_state_dict_d)

    if saved_state_dict_g is not None and starting_episode is not None:
        generator_model.load_state_dict(saved_state_dict_g)
    else:
        starting_episode = -1

    # Rollout
    rollout = Rollout(generator_model, config.ROLLOUT_UPDATE_RATE, device)

    # losses
    criterion_d = nn.BCELoss()
    criterion_g_1 = nn.CrossEntropyLoss()
    gen_gan_loss = GANLoss(device)

    # Optimizers
    optimizer_d = optim.Adam(discriminator_model.parameters(), lr=config.LEARNING_RATE_D)
    optimizer_g = optim.Adam(generator_model.parameters(), lr=config.LEARNING_RATE_G)

    if is_demo:
        manual_test_lines = []
        for each_line in config.MANUAL_TEST_LINES:
            manual_test_lines.append(tdata.convert_line_to_indexes(each_line, char_array))
        manual_test_lines = torch.LongTensor(np.array(manual_test_lines)).to(device=device)
        valuate_generator(generator_model, manual_test_lines, char_array, None)
    else:
        print('Starting epoch:{}'.format(starting_episode + 1))
        for e in range(starting_episode + 1, config.EPOCH):
            d_count = 0
            total_loss_d = 0
            g_count = 0
            total_loss_g = 0
            considered_real = 0

            for i, (real_poem1_idx, real_poem2_idx) in enumerate(tqdm(train_loader)):
                if real_poem2_idx.shape[0] != config.BATCH_SIZE:
                    break

                '''
                random_input = generate_random_poems(batch_size=config.BATCH_SIZE,
                                                     char_size=char_array.size,
                                                     start_idx=2, end_idx=3, pad_idx=0,
                                                     dvc=device)
                '''
                # This output is to be used in the GAN training.
                # Generator's output is in (batch * sequence * embedding dim)
                g_output_1 = generator_model(real_poem1_idx)

                # Labels
                label_true = torch.ones(config.BATCH_SIZE, dtype=torch.float32, device=device)
                label_false = torch.zeros(config.BATCH_SIZE, dtype=torch.float32, device=device)

                d_input_1 = torch.cat((real_poem2_idx, g_output_1.detach().argmax(2)), dim=0)
                d_label = torch.cat((label_true, label_false), dim=0)

                d_output_1 = discriminator_model(d_input_1).squeeze()
                loss_d = criterion_d(d_output_1, d_label)

                # Skip training the discriminator when it is too strong.
                wrong_judge_value = (d_output_1 * torch.cat((label_false, label_true), dim=0)).mean() * 2
                if d_count == 0 or wrong_judge_value.item() > config.D_TRAIN_THRESHOLD:
                    d_count += 1
                    total_loss_d += loss_d
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator_model.parameters(), config.CLIP)
                    optimizer_d.step()

                # Train the generator
                g_count += 1
                if e < config.EPOCH * (1 - config.GAN_RATE_IDX):
                    # Learn training samples.
                    g_output_2 = generator_model(real_poem1_idx, real_poem2_idx,
                                                 teacher_forcing_ratio=config.TEACHER_FORCING_RATE)
                    # g_output_2 is in (batch, sequence, embedding)

                    g_output_2 = g_output_2.reshape(g_output_2.shape[0] * g_output_2.shape[1],
                                                    g_output_2.shape[2]).contiguous()
                    tar = real_poem2_idx.view(-1)

                    loss_g = criterion_g_1(g_output_2, tar)
                    total_loss_g += loss_g
                    optimizer_g.zero_grad()
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(generator_model.parameters(), config.CLIP)
                    optimizer_g.step()

                else:
                    # Train as GAN
                    g_output_samples = generator_model.sample(real_poem1_idx, config.BATCH_SIZE,
                                                              real_poem1_idx.shape[1])
                    rewards = rollout.get_reward(g_output_samples, config.ROLLOUT_NUMBER, discriminator_model,
                                                 real_poem1_idx)
                    considered_real += rewards.detach().mean().cpu().item()
                    prob = generator_model(real_poem1_idx, real_poem2_idx, teacher_forcing_ratio=1)

                    '''
                        Reshaping:
                            Before reshaping:
                                prob in (batch_size, seq_len, char_size)
                                g_output_samples in (batch_size, seq_len)
                                rewards in (batch_size, seq_len)
                            Shape of the inputs:
                                prob in (batch_size * seq_len, char_size)
                                g_output_samples in (batch_size * seq_len)
                                rewards in (batch_size * seq_len)
                    '''
                    prob = prob.reshape(prob.shape[0] * prob.shape[1], prob.shape[2]).contiguous()
                    g_output_samples = g_output_samples.reshape(
                        g_output_samples.shape[0] * g_output_samples.shape[1]).contiguous()
                    rewards = rewards.reshape(rewards.shape[0] * rewards.shape[1]).contiguous()

                    loss_g = gen_gan_loss(prob, g_output_samples, rewards)
                    loss_g = loss_g / config.BATCH_SIZE
                    total_loss_g += loss_g
                    optimizer_g.zero_grad()
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(generator_model.parameters(), config.CLIP)
                    optimizer_g.step()

            # Update the generator model in rollout.
            rollout.update_module()
            # Save the model in the end of each epoch.
            save_loader_d.save_module(discriminator_model, e)
            save_loader_g.save_module(generator_model, e)

            if g_count == 0:
                print('Epoch {}: avg_loss_d: {:.4f}'.format(e, total_loss_d / d_count))
            else:
                print('Epoch {}: avg_loss_d: {:.4f}, avg_loss_g: {:.4f}, considered_real: {:.4f}'.format(e,
                                                                                                         total_loss_d / d_count,
                                                                                                         total_loss_g / g_count,
                                                                                                         considered_real / g_count))

            manual_test_lines = []
            for each_line in config.MANUAL_TEST_LINES:
                manual_test_lines.append(tdata.convert_line_to_indexes(each_line, char_array))
            manual_test_lines = torch.LongTensor(np.array(manual_test_lines)).to(device=device)
            valuate_generator(generator_model, manual_test_lines, char_array, None)
            valuate_generator(generator_model, manual_test_lines, char_array, None)


if __name__ == "__main__":
    train_4(False)
