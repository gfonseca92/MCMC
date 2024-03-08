import numpy as np

X = np.linspace(0., 11., 100000)


class MyDistribution:

    def pdf(self, x):
        # return np.sin(x)
        return 1.5 * self.norm(x, 3., 1) + 3.5 * self.norm(x, 8., 1)

    def norm(self, x, mu, sigma):
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x - mu)**2 / (2*sigma**2))

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def rvs(self):
        current_x = np.random.choice(X)
        return self.pdf(current_x)

