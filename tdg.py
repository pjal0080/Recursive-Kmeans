import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans



df1 = pd.read_csv("data.csv")
df2 = pd.read_csv("labels.csv")



y = df2.iloc[:,1].values


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
y = label.fit_transform(y)

cs = []

for i in range(len(y)):
    cs.append(y[i])


sub = pd.DataFrame(cs)
sub.to_csv("newlabel.csv", index = False)

'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


wcss = []

for i in range(1, 11):
    kmean = KMeans(n_clusters = i, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
    kmean.fit(X)
    wcss.append(kmean.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')      #within cluster sum of squares
plt.show()


'''



















