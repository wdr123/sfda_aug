import seaborn as sns # for data visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

df_6row = [36.40,36.50,36.71,36.68,36.62,36.60,36.54]
df_7row = [36.83,37.14,37.55,37.18,36.74,36.72,36.70]
df_8row = [36.95,37.28,37.84,37.25,36.88,36.85,36.85]
df_9row = [36.85,37.16,37.59,37.22,36.78,36.75,36.73]
df_10row = [36.78,37.06,37.49,37.09,36.70,36.67,36.65]
df_11row = [36.68,36.92,37.35,36.98,36.65,36.61,36.59]
df_12row = [36.53,36.79,37.20,36.84,36.51,36.47,36.46]

df = [df_6row,df_7row,df_8row,df_9row,df_10row,df_11row,df_12row]


x_axis_labels = [6,7,8,9,10,11,12] # labels for x-axis
y_axis_labels = [6,7,8,9,10,11,12] # labels for y-axis

# create seabvorn heatmap with required labels
s = sns.heatmap(df, xticklabels=x_axis_labels, yticklabels=y_axis_labels)

s.set_xlabel('M1', fontsize=15)
s.set_ylabel('M2', fontsize=15)

plt.title("Hyper-parameter M1/M2 length ablation.", fontsize = 15)
plt.savefig('imgs/2dheatmap.png')
plt.close()
