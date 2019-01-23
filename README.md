# Polynomial Kernel Random Projection

Implementation of Random Projection for homogeneous polynomial kernel feature spaces, introduced in [1]. This is a Python implementation, accelerated using [numba](http://numba.pydata.org/). The following pseudo-code describes the implemented algorithm. For a more high-level description of the algorithm, please refer to the original publication.

<p align="center">
<img src="https://github.com/lopeLH/PolyKernel-RP/blob/master/repo_images/algorithm.png" width="700" />
</p>

Here you can see the evolution of distance preservation as the size of the output space (n_components) grows. This was generated with p=5000, t=20, degree=2.

<p align="center">
<img src="https://github.com/lopeLH/PolyKernel-RP/blob/master/repo_images/fire.gif" width="700" />
</p>

[1] [López-Sánchez, D., Arrieta, A. G., & Corchado, J. M. (2018). Data-independent Random Projections from the feature-space of the Homogeneous Polynomial Kernel. Pattern Recognition.](https://www.sciencedirect.com/science/article/pii/S0031320318301675)
