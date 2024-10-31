import numpy as np

from scatter_plot import ScatterPlot

np.random.seed(42)
hamming = np.random.rand(1000) * 10
approvalwise = hamming + (np.random.randn(1000) * 0.005)

plot = ScatterPlot(hamming, approvalwise, "Hamming distance", "Approvalwise distance")
plot.plot(show=True)
 