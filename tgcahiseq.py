import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


df1 = pd.read_csv("data.csv")
df2 = pd.read_csv("newlabel.csv")
df1 =  df1.drop(['Unnamed: 0'],axis = 1)
df = pd.concat([df1,df2],axis = 1)

#MEDIAN NORMALIZATION AND VARIANCE STABALIZATION
c = pd.DataFrame(df1.sum(axis = 1),columns = ['Count'])
C = c.median()
c = np.array(c)
train = df1.transpose()
train.head()

x = []
for i in range(801):
    x.append(C/c[i])

for i in range(0,801):
    train[i] = train[i].map(lambda k : k * x[i])
    train[i] = train[i].map(lambda k : 2 * math.sqrt(k))


#pca
X_train = train.transpose()
X_train.head(10)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
ex_var = pca.explained_variance_ratio_






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





def dist_func(x,y):
    
    ans = 0
    for i in range(len(x)-1):
        ans += pow(x[i] - y[i],2)
        
    return math.sqrt(ans)




def kmeans(ds,centroid,k):
    
    max_iter = 100
    
    for i in range(max_iter):
        
        d = {}
        for i in range(k):
            d[i]=[]
            
        d1 = {}
        for i in range(k):
            d1[i]=[]
            
        for j in range(len(ds)):
            l= []
                
            for m in centroid:
                
                l.append(dist_func(m,ds[j]))
                
            d[l.index(min(l))].append(ds[j])
            
            
        
        for z in d:
            centroid[z] = np.average(d[z],axis = 0)
            
        
        
    return d




def rec_kmeans(ds):
    
    
    li = set()
        
    for j in ds:
        
        if(j[2] != -1):
            li.add(j[2])
        
        
    
    if(len(li) == 1):
        final_clusters.append(ds)
        cluster_label.append(li)
        return

    
    if(len(li) == 0):
        final_clusters.append(ds)
        cluster_label.append(-1)
        return
    
    
    
            
    
    newk = len(li)
    centroid = []
    
    for i in range(newk):
        centroid.append(ds[i])
    
    r = kmeans(ds,centroid,newk)
    
    for z in r:
        z1 = r[z]
        rec_kmeans(z1)
    



k = 5

X_train = pd.DataFrame(X_train)
y = df['Label']
X1 = pd.concat([X_train,y],axis = 1)
X_training = X1.iloc[:,[0,1,2]].values

final_clusters = []
cluster_label = []
centr = []
for i in range(k):
        centr.append(X_training[i])
    


#initial Kmeans call  
res = kmeans(X_training,centr,k)


colors = ["r", "g", "c", "b", "k"]
labl = [0, 1, 2, 3, 4]



for c in res:
	color = colors[c]
	for features in res[c]:
		plt.scatter(features[0], features[1], color = color ,s = 30)


for c in centr:
    plt.scatter(c[0],c[1],marker = 'X',s = 150 ,color = 'm')
        

plt.show()





#recursive Kmeans call for every partition in res
for par in res:
    pi = res[par]
    rec_kmeans(pi)



final_centroids = []


#calculating final centroids
for i in final_clusters:
    final_centroids.append(np.average(i,axis = 0))



#showing results
results = pd.DataFrame(dtf,columns = ['Clusters'])
no_of_points = []
labelled = []
unlabelled = []

cluster_lbl = []



for i in cluster_label:
    if( i != -1):
        cluster_lbl.append(int(list(i)[0]))
    else:
        cluster_lbl.append(i)

results['Label'] = cluster_lbl  



for i in range(len(final_clusters)):
    no_of_points.append(len(final_clusters[i]))
    cl = 0
    cu = 0
    for j in final_clusters[i]:
        if(j[2] == -1):
            cu += 1
        else:
            cl += 1
    
    labelled.append(cl)
    unlabelled.append(cu)
    
results['Total_no_of_data'] = no_of_points
results['No_of_ld'] = labelled
results['No_of_ud'] = unlabelled

print(results)


