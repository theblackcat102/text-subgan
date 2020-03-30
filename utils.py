import os
import pickle
import numpy as np
import torch
import torch.autograd as autograd

from preprocess import clean_text, segment_text, pad_sequence
from constant import Constants, CACHE_DIR, WRITER_PATTERN


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
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad = True
    disc_interpolates = netD(interpolates)

    ones = torch.ones(disc_interpolates.size())
    if torch.cuda.is_available():
        ones = ones.cuda()

    gradients = autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty


def transfer(E, G, text, from_style, to_style, max_length, device, mapping={}):
    """
    Args:
        text: str
        from_style: 'ptt' or 'dcard'
        from_style: 'ptt' or 'dcard'
        max_length: int
        device: torch device
    Returns
        text: str
    """
    E.eval()
    G.eval()
    with torch.no_grad():
        if from_style == 'ptt':
            from_style = torch.zeros(1).long().to(device)
        else:
            from_style = torch.ones(1).long().to(device)
        if to_style == 'ptt':
            to_style = torch.zeros(1).long().to(device)
        else:
            to_style = torch.ones(1).long().to(device)
        seq = pad_sequence(Voc.encode(text), max_length)
        seq = torch.tensor(seq).long().to(device).unsqueeze(0)
        outputs, latent, _, _ = E(seq, from_style, device)
        outputs = G(latent, to_style, outputs, max_length, device)
        outputs = torch.argmax(outputs, dim=-1)
        outputs = outputs.detach().cpu().numpy()
        outputs = Voc.decode(outputs[0], mapping=mapping)
    E.train()
    G.train()
    return outputs


def write_example(E, G, batch_x, batch_y, writer, num_iter, max_length,
                  device):
    ptt2ptt = []
    ptt2dcard = []
    mapping = Constants.idx2example()
    for seq_x in batch_x['seq'].numpy():
        real_x = Voc.decode(seq_x, mapping=mapping)
        rect_x = transfer(E, G, real_x, 'ptt', 'ptt', max_length, device)
        fake_y = transfer(E, G, real_x, 'ptt', 'dcard', max_length, device)
        real_x = Voc.decode(seq_x)
        ptt2ptt.append(WRITER_PATTERN % (real_x, rect_x))
        ptt2dcard.append(WRITER_PATTERN % (real_x, fake_y))
    writer.add_text('PTT_to_PTT', '\n'.join(ptt2ptt), num_iter)
    writer.add_text('PTT_to_DCARD', '\n'.join(ptt2dcard), num_iter)

    dcard2dcard = []
    dcard2ptt = []
    for seq_y in batch_y['seq'].numpy():
        real_y = Voc.decode(seq_y, mapping=mapping)
        rect_y = transfer(E, G, real_y, 'dcard', 'dcard', max_length, device)
        fake_x = transfer(E, G, real_y, 'dcard', 'ptt', max_length, device)
        real_y = Voc.decode(seq_y)
        dcard2dcard.append(WRITER_PATTERN % (real_y, rect_y))
        dcard2ptt.append(WRITER_PATTERN % (real_y, fake_x))
    writer.add_text('DCARD_to_DCARD', '\n'.join(dcard2dcard), num_iter)
    writer.add_text('DCARD_to_PTT', '\n'.join(dcard2ptt), num_iter)


# class Voc:
#     idx2word = pickle.load(open(os.path.join(CACHE_DIR, "idx2word.pkl"), 'rb'))
#     word2idx = pickle.load(open(os.path.join(CACHE_DIR, "word2idx.pkl"), 'rb'))
#     vocab_size = len(idx2word)

#     @classmethod
#     def encode(cls, words):
#         if len(words) == 0:
#             return []
#         cleaned_text = clean_text(words)
#         segmented = segment_text(cleaned_text)
#         ret = []
#         for word in segmented:
#             if word in cls.word2idx:
#                 ret.append(cls.word2idx[word])
#             else:
#                 ret.append(cls.word2idx[Constants.UNK_WORD])
#         # return [Constants.BOS] + ret + [Constants.EOS]
#         return np.array(ret)

#     @classmethod
#     def decode(cls, idxs, stop=Constants.EOS, mapping={}):
#         ret = ""
#         for idx in idxs:
#             if idx in cls.idx2word:
#                 if idx in mapping:
#                     w = mapping[idx]
#                 else:
#                     w = cls.idx2word[idx]
#             else:
#                 w = Constants.UNK_WORD
#             ret += w
#             if stop is not None and idx == stop:
#                 break
#         return ret