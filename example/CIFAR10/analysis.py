import numpy as np
import matplotlib.pyplot as plt

# load in the .npy file

file = "./SHAP/data_bird.npy"
shap = "./SHAP/shap_bird_airplane.npy"

data = np.load(file)
shap = np.load(shap)

# plot with cmap red/blue

plt.imshow(data/data.max(), cmap='gray', alpha=0.2)
plt.imshow(shap/shap.max(), cmap='bwr', alpha=.7)
plt.colorbar()
plt.savefig('./SHAP/SHAP_bird_airplane.png')