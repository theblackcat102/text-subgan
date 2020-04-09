import pickle
import os
import torch
from collections import Counter

# path = 'save/sub_relgan-mini-kmeans-2020-04-08-20-30-00' # kmeans predict do not fit after update eigenvectors

checkpoint = torch.load(os.path.join(path, 'relgan_G_2000.pt'))

print(Counter(checkpoint['p']))
print(checkpoint['latent'][0])


checkpoint = torch.load(os.path.join(path, 'relgan_G_6000.pt'))

print(Counter(checkpoint['p']))
print(checkpoint['latent'][0])


checkpoint = torch.load(os.path.join(path, 'relgan_G_10000.pt'))
print(Counter(checkpoint['p']))
print(checkpoint['latent'][0])
