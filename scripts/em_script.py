import numpy as np
from matplotlib import pyplot as plt
from src.em import EM


if __name__ == '__main__':
    samples = np.load('../data/samples.npz')
    X = samples['data']
    pi0 = samples['pi0']
    mu0 = samples['mu0']
    sigma0 = samples['sigma0']
    em = EM(X)
    # gamma = em._e_step(pi0, mu0, sigma0)
    # print(gamma.sum(0))
    # pi, mu, sigma = em._m_step(gamma)
    # print(pi)
    # print(mu)
    # print(sigma)
    # loss = em._compute_vlb(pi, mu, sigma, gamma)
    em.fit(C=3)
    labels = em.transform().argmax(axis=1)
    colors = np.array([(31, 119, 180), (255, 127, 14), (44, 160, 44)]) / 255.
    plt.scatter(em.x[:, 0], em.x[:, 1], c=colors[labels], s=30)
    plt.axis('equal')
    plt.show()
