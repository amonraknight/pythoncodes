# This util should have the functions to evaluate models.
import torch
import random
import math


# val: {m, n} ndarray
def translate_batch_to_char(val, char_array):
    for each_row in val:
        print('prediction {}'.format(idx2char(char_array, each_row)))


def translate_src_tgt_to_char(src, tgt, char_array):
    if len(src) != len(tgt):
        print('Source {} and target {} are not of the same size.'.format(len(src), len(tgt)))
    else:
        for row in range(len(src)):
            print('source {} , prediction {}'.format(idx2char(char_array, src[row]), idx2char(char_array, tgt[row])))



def valuate_generator(model, val_x, char_array, val_y, batch_first=True):
    # test
    output_array = [None, None, None]
    v_output = val_x

    for i in range(0, 3):

        # for sequence-2-sequence
        # v_output = model(v_output, None, max_gen_length=9)
        # for transformer
        with torch.no_grad():
            if not batch_first:
                v_output = v_output.transpose(0, 1)
            v_output = model(v_output)

        # v_output (batch, sequence_size, char_size) -- > to index in (batch, sequence_size)
        v_output = v_output.argmax(-1)
        output_array[i] = v_output.detach().cpu().numpy()

    for j in range(val_x.shape[0]):
        src_text = idx2char(char_array, val_x[j])
        if val_y is None:
            tar_text = ''
        else:
            tar_text = idx2char(char_array, val_y[j])

        predit_text = ''
        for i in range(0, 3):
            predit_text = predit_text + ' ' + idx2char(char_array, output_array[i][j])
        print('context line: {}, target {}, prediction {}'.format(src_text, tar_text, predit_text))


def translate_result(output, char_array):
    output = output.argmax(-1)
    output_array = output.detach().cpu().numpy()
    translate_batch_to_char(output_array, char_array)


# The generator model produces embedded outputs in (batch * sequence * embedding dim).
def valuate_generator2(model, val_x, embedder, char_array):
    # test
    output_array = [None, None, None]
    v_output = val_x
    for i in range(0, 3):

        # for sequence-2-sequence
        v_output = model(v_output)

        # v_output (batch, sequence_size, embedding dim) -- > to index in (batch, sequence_size)
        v_output = embedder.find_closet_vector_idx(v_output)
        output_array[i] = v_output.detach().cpu().numpy()

    for j in range(val_x.shape[0]):
        src_text = idx2char(char_array, val_x[j])
        predit_text = ''
        for i in range(0, 3):
            predit_text = predit_text + ' ' + idx2char(char_array, output_array[i][j])
        print('context line: {}, prediction {}'.format(src_text, predit_text))


def idx2char(char_array, idx_list):
    text = []
    for i in idx_list:
        text.append(char_array[i])
    return ''.join(text)


# ['P', 'U', 'S', 'E']
def generate_random_poems(batch_size, char_size, start_idx, end_idx, pad_idx, dvc):
    shape = (batch_size, 9)
    random_integers_tensor = torch.randint(low=max(start_idx, end_idx, pad_idx),
                                           high=char_size-1, size=shape)

    random_integers_tensor[:, 0] = start_idx
    random_integers_tensor[:, 8] = end_idx

    random_list = [random.randint(0, batch_size-1) for i in range(math.floor(batch_size/2))]

    # Some set to 5 chars.
    random_integers_tensor[random_list, 6] = end_idx
    random_integers_tensor[random_list, 7] = pad_idx
    random_integers_tensor[random_list, 8] = pad_idx

    return random_integers_tensor.to(dvc)


def require_grad(model):
    for param in model.parameters():
        param.requires_grad = True


def watch_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(name, param.grad)


if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    result = generate_random_poems(8, 10000, 2, 3, 0, device)
    print(result)
