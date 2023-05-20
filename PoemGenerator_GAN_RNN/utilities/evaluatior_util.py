# This util should have the functions to evaluate models.
import torch
import random
import math


def valuate_generator(model, val_x, char_array, val_y):
    # test
    output_array = [None, None, None]
    v_output = val_x
    for i in range(0, 3):
        v_output = model(v_output, None, teacher_forcing_ratio=0, max_gen_length=9)
        # v_output (batch, sequence_size, char_size) -- > to index in (batch, sequence_size)
        v_output = v_output.argmax(2)
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




if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    result = generate_random_poems(8, 10000, 2, 3, 0, device)
    print(result)
