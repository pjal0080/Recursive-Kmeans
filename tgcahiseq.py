import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
from collections import Counter


df1 = pd.read_csv("data.csv")
df2 = pd.read_csv("newlabel.csv")
df = pd.concat([df1,df2],axis = 1)


X = df.iloc[:,[i for i in range(1,20532)]].values
y = df.iloc[:,-1].values



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)
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

X = pd.DataFrame(X)
y = pd.DataFrame(y)
X1 = pd.concat([X,y],axis = 1)

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



#ploting final results
for fea in final_clusters:
    color = 'k'
    for k in fea:
        plt.scatter(k[0], k[1],color = color,s = 100)



for cen in final_centroids:
    plt.scatter(cen[0],cen[1],marker = "X",color = "m",s = 10)
    
   
plt.show()




