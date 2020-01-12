import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from collections import Counter



df1 = pd.read_csv("iris.csv")
df2 = pd.read_csv("label.csv")
df = pd.concat([df1,df2],axis = 1)


df.describe()
df.info()
df = df.drop(['label'], axis = 1)
corrm = df.corr()
sns.heatmap(corrm,vmax = .8, square = True)



X = df.iloc[:,[0,2,3]].values
y = df.iloc[:,-1].values


'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
'''


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
        
        if(j[3] != -1):
            li.add(j[3])
        
        
    
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
        
 



       
k = 3

X = pd.DataFrame(X)
y = pd.DataFrame(y)
X1 = pd.concat([X,y],axis = 1)

X_training = X1.iloc[:,0:4].values

final_clusters = []
cluster_label = []
centr = []
for i in range(k):
        centr.append(X_training[i])


#initial Kmeans call  
res = kmeans(X_training,centr,k)


for i in res:
    ct0 = 0
    ct1 = 0
    ct2 = 0
    tot = 0
    for j in res[i]:
        tot = len(res[i])
        if(j[3] == 1):
            ct1 += 1
        elif(j[3] == 2):
            ct2 += 1
        elif(j[3] == 0):
            ct0 += 1
    
    per0 = 0
    per1 = 0
    per2 = 0
        
    if(ct0 != 0):
        per0 = (ct0/tot) * 100
    if(ct1 != 0):
        per1 = (ct1/tot) * 100
    if(ct2 != 0):
        per2 = (ct2/tot) * 100
        
    print(per0,per1,per2)
    
    
thers = 6     
            

colors = ["r", "g", "c", "b", "k"]



for c in res:
	color = colors[c]
	for features in res[c]:
		plt.scatter(features[0], features[1], color = color,s = 30)


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
    for k in fea:
        plt.scatter(k[0], k[1], color = "k",s = 10)



for cen in final_centroids:
    plt.scatter(cen[0],cen[1],marker = "X",color = "m",s = 30)
    
   
plt.show()
  