import numpy as np
import matplotlib.pyplot as plt

# load in the .npy file

file = "./SHAP/0_original.npy"
shap = "./SHAP/0_shap_8.npy"

data = np.load(file)
shap = np.load(shap)
data = data.reshape(28,28)
shap = shap.reshape(28,28)

# plot with cmap red/blue
plt.imshow(data/data.max(), cmap='gray', alpha=0.5)
plt.imshow(shap/shap.max(), cmap='bwr', alpha=.5)
plt.colorbar()
plt.savefig('./SHAP/0_SHAP_8.png')