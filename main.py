import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from hmc import hmc
from nuts import nuts
from custom_dist import MyDistribution, X

# x, y = np.mgrid[-1.5:1.5:.01, -1:1.5:.01]
# pos = np.dstack((x, y))
# rv = st.multivariate_normal([0.0, 0.5], [[2.0, 0.0], [0.0, 0.5]])
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# ax2.contourf(x, y, rv.pdf(pos))


true_dist = MyDistribution() #st.norm(0, 1)#st.beta(5.567, 1.789)#
plt.figure(figsize=(4, 4))
plt.plot(X, true_dist.pdf(X), 'r')
# x = np.linspace(0, 1, 500)
# plt.plot(x, true_dist.pdf(x), 'r')

samples, H = hmc(true_dist, 10., 0.1, 10000)
nuts(true_dist, 10., 0.1, 10000)

# plt.hist(samples, bins=50, density=True)
# plt.show()
