import torch
import utils

# find letter index of all letters
def letterToIndex(letter):
    return utils.all_letters.find(letter)

# demonstration : turning a letter into 1 x n_letters Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, utils.num_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# turn a line into 1 x n_letter x line_length
# or an array of one hot letter vectors

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, utils.num_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# print(letterToTensor("J"))
# # print(lineToTensor("Jones"))
# print(lineToTensor("Jones").size())


