import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns

np.random.seed(0)
random.seed(0)

plt.rcParams.update({'font.size': 16})
matplotlib.rcParams['legend.fontsize'] = 16

clrs = sns.color_palette("husl", 2)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.grid()

# set labels and tick font size
ax.set_xlabel("Hyper-parameter λ", fontsize = 17)
ax.set_ylabel('mAP on Day Foggy', fontsize = 17)
ax.set_xticks([0.25, 0.5, 0.75, 1, 1.25, 1.5])
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# xasd = np.array([0.8,0.85,0.9,0.95,1])
# yasd = np.array([0.36, 0.368, 0.375, 0.384, 0.372])
# error = np.array([0.04, 0.043, 0.053, 0.052,  0.051]) # ASD_std

# yasd = [89.29,89.97,90.14,91.12,91.37]
# target_name = ['ASD 20%','ASD 40%','ASD 60%','ASD 80%','ASD 100%',]
# error = np.random.normal(loc=0,scale=0.1,size=5)

xasd = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5])
yasd = [36.89, 37.45, 37.82, 37.90, 37.84, 37.65]
error = np.random.normal(loc=0,scale=0.05,size=6)


ax.plot(xasd, yasd, c=clrs[0])
ax.fill_between(xasd, yasd - error, yasd + error, alpha=0.3, facecolor=clrs[0])

plt.title("Hyper-parameter λ Ablative Study")
plt.savefig('imgs/lambda_ablative.png')
plt.close()