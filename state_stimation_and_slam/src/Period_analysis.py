import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../../../results/Slam2.csv')

print(df.columns[0])
header = df.columns[0]

# block 1 - simple stats
mean1 = df[header].mean()
sum1 = df[header].sum()
max1 = df[header].max()
min1 = df[header].min()
count1 = df[header].count()
median1 = df[header].median()
std1 = df[header].std()
var1 = df[header].var()

# print block 1
print ('Mean period: ' + str(mean1) + ' s')
print ('Sampling time: ' + str(sum1) + ' s')
print ('Max period: ' + str(max1) + ' s')
print ('Min period: ' + str(min1) + ' s')
print ('Number of samples: ' + str(count1))
print ('Median period: ' + str(median1) + ' s')
print ('Std of periods: ' + str(std1))
print ('Var of periods: ' + str(var1))

x = np.array(range(count1))
z = np.polyfit(x, np.array(df[header]), 4)
y = np.poly1d(z)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(header)
ax.set_ylabel('Period (s)')
ax.set_xlabel('Sampling time (s)')

plt.plot(x, y(x), '.', x, df[header])
plt.show()
