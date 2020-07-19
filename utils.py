import os
import pickle
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from preprocess import clean_text, segment_text, pad_sequence
from constant import Constants, CACHE_DIR, WRITER_PATTERN


def data_iter(dataloader):
    def function():
        while True:
            for batch in dataloader:
                yield batch
    return function()


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_pretrain_embedding():
    return pickle.load(open(os.path.join(CACHE_DIR, "embedding.pkl"), 'rb'))


def variable_mask(length, max_length, front=True):
    mask = torch.arange(max_length).unsqueeze(0).expand(len(length), -1)
    if torch.cuda.is_available():
        mask = mask.cuda()
    b = length.unsqueeze(1).expand(-1, max_length)
    if front:
        return mask < b
    else:
        return mask >= b

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    try:
        length = len(lst)
    except TypeError:
        length = lst.shape[0]
    for i in range(0, length, n):
        yield lst[i:i + n]

def one_hot(index, depth):
    size = index.size()
    index = index.flatten()
    one_hot_tensor = torch.zeros(index.size(0), depth)
    one_hot_tensor[torch.arange(index.size(0)), index] = 1.
    if torch.cuda.is_available():
        one_hot_tensor = one_hot_tensor.cuda()
    return one_hot_tensor.view(*size, -1).detach()


def straight_through_estimate(p):
    shape = p.size()
    ind = p.argmax(dim=-1)
    p_hard = torch.zeros_like(p).view(-1, shape[-1])
    p_hard.scatter_(1, ind.view(-1, 1), 1)
    p_hard = p_hard.view(*shape)
    return ((p_hard - p).detach() + p)


def sample_gumbel(shape, eps=1e-20):
    u = torch.rand(shape)
    if torch.cuda.is_available():
        u = u.cuda()
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax(logits, temperature, st_mode=False):
    """
    Gumble Softmax

    Args:
        logits: float tensor, shape = [*, n_class]
        temperature: float
        st_mode: boolean, Straight Through mode
    Returns:
        return: gumbel softmax, shape = [*, n_class]
    """
    logits = logits + sample_gumbel(logits.size())
    return softmax(logits, temperature, st_mode)


def softmax(logits, temperature=1, st_mode=False):
    """
    Softmax

    Args:
        logits: float tensor, shape = [*, n_class]
        st_mode: boolean, Straight Through mode
    Returns:
        return: gumbel softmax, shape = [*, n_class]
    """
    y = torch.nn.functional.softmax(logits, dim=-1)
    if st_mode:
        return straight_through_estimate(y)
    else:
        return y

def binary_accuracy(preds, y, logits=True):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8,
    NOT 8
    """
    # round predictions to the closest integer
    if logits:
        rounded_preds = torch.round(torch.sigmoid(preds))
    else:
        rounded_preds = torch.round(preds)
    rounded_target = torch.round(y)
    # convert into float for division
    correct = (rounded_preds == rounded_target).float()
    acc = correct.sum() / len(correct)
    return acc.cpu().item()


def gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1)
    alpha = alpha.expand(real_data.size())
    # print(alpha.shape, real_data.shape, fake_data.shape)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad = True
    disc_interpolates = netD(interpolates)
    if isinstance(disc_interpolates, tuple):
        disc_interpolates = disc_interpolates[0]

    ones = torch.ones(disc_interpolates.size())
    if torch.cuda.is_available():
        ones = ones.cuda()

    gradients = autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
    gradient_penalty = ((gradients_norm - 0) ** 2).mean()
    return gradient_penalty



def truncated_normal_(tensor, mean=0, std=1):
    """
    Implemented by @ruotianluo
    See https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def get_fixed_temperature(temper, i, N, adapt):
    """A function to set up different temperature control policies"""
    # N = 5000

    if adapt == 'no':
        temper_var_np = 1.0  # no increase, origin: temper
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1) ** 2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        raise Exception("Unknown adapt type!")

    return temper_var_np


def get_losses(d_out_real, d_out_fake, loss_type='JS'):
    """Get different adversarial losses according to given loss_type"""
    bce_loss = nn.BCEWithLogitsLoss()

    if loss_type == 'standard':  # the non-satuating GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = bce_loss(d_out_fake, torch.ones_like(d_out_fake))

    elif loss_type == 'JS':  # the vanilla GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake

    elif loss_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = torch.mean(-d_out_fake)

    elif loss_type == 'wasstestein':
        d_loss = d_out_fake.mean() - d_out_real.mean()
        g_loss = -d_out_fake.mean()

    elif loss_type == 'hinge':  # the hinge loss
        d_loss_real = torch.mean(nn.ReLU(1.0 - d_out_real))
        d_loss_fake = torch.mean(nn.ReLU(1.0 + d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -torch.mean(d_out_fake)

    elif loss_type == 'tv':  # the total variation distance
        d_loss = torch.mean(nn.Tanh(d_out_fake) - nn.Tanh(d_out_real))
        g_loss = torch.mean(-nn.Tanh(d_out_fake))

    elif loss_type == 'rsgan':  # relativistic standard GAN
        d_loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real))
        g_loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % loss_type)

    return g_loss, d_loss
