# Recursive-Kmeans

In this project , we were given the task to apply recursive clustering on single cell datasets that have some labelled 
and some unlabelled data such that each cluster contains labelled data from same class.
For example , suppose initially I have 5 clusters. Each cluster contains some labelled(suppose 0 to 4) and some 
unlabelled data(represented by -1).If a cluster contains labelled data from more than one class then we do recursive KMeans 
with k being equal to the total no of labels present in the cluster until it is divided into clusters that contains 
labelled data from a single class.We do this for all initial clusters one by one.The unlabelled data is then given the label 
of the cluster in which it is present. 

Here ,I have performed recursive KMeans on two datasets,first one is the Iris dataset and 
second one is gene expression cancer RNA-Seq Data Set(link : https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq#).
The results can be visualised on the notebooks thats been uploaded and pure code can be seen on the .py files.

