import torch


def Batch_Sample(data, label):
    index = torch.randint(low=0, high=data.shape[0], size=label.shape)
    sampled_data = torch.zeros((len(label), data.shape[2], data.shape[3], data.shape[4])).cuda().half()
    for i in range(len(label)):
        sampled_data[i, :, :, :] = data[index[i], label[i], :, :, :]
    return sampled_data