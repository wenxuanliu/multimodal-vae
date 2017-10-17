from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from generator import ShuffleMNIST
from train import load_checkpoint

from sklearn.decomposition import PCA


if __name__ == "__main__":
    import os
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    loader = torch.utils.data.DataLoader(
        ShuffleMNIST('./data/processed/test.pt',
                     transform=transforms.ToTensor()),
        batch_size=128, shuffle=True)

    vae = load_checkpoint('./trained_models/model_best.pth.tar', use_cuda=args.cuda)
    vae.eval()

    for batch_idx, (image, text) in enumerate(loader):
        if args.cuda:
            image, text = image.cuda(), text.cuda()
        image, text = Variable(image, volatile=True), Variable(text, volatile=True)
        image = image.view(-1, 784)  # flatten image

        z = vae.gen_latents(image, text)
        if batch_idx == 0:
            latents = z
            labels = text
        else:
            latents = torch.stack((latents, z))
            labels = torch.stack((labels, text))

    latents = latents.cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    if latents.shape[1] > 2:
        pca = PCA(n_components=2)
        pca.fit_transform(latents)

    # now we have latents guaranteed to be 2 dimensions
    # let's plot the manifold
    colors = iter(cm.rainbow(np.linspace(0, 1, 10)))  # ten kinds of colors

    plt.figure()
    for i in xrange(10):
        latents_i = latents[labels == i]
        plt.scatter(latents_i[0], latents_i[1], color=next(colors), 
                    label=str(i), alpha=0.3, edgecolors='none')
    plt.legend()
    if not os.path.exists('./results'):
        os.makedirs('./results')
    plt.savefig('./results/manifold.png')
