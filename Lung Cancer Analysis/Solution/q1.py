#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import sys


# In[2]:


#output redirection
sys.stdout=open('output_1.txt','w')


# In[3]:


#reading the data
data = pd.read_csv('lung-cancer.data',header=None)
data.head()


# In[4]:


#replacing '?' with median values
data = data.apply(pd.to_numeric,errors='coerce')
data.fillna(data.median(numeric_only=True).round(1),inplace=True)
data.head()


# In[5]:


# finding the entropy of classes (will be used in calculating NMI)
c1= (data[0].value_counts()[1])/data.shape[0]  # number of data elements having class 1
c2= (data[0].value_counts()[2])/data.shape[0]  # number of data elements having class 2
c3= (data[0].value_counts()[3])/data.shape[0]  # number of data elements having class 3
entropy_classes = (-c1)*(np.log(c1))-(c2)*(np.log(c2))-(c3)*(np.log(c3))
# print(entropy_classes)


# In[6]:


class_list=[]
class_list.extend(data[0].tolist())
print("PCA")
print("\n")
print("# Following is the column in data that shows the type of cancer = \n",class_list)
print("\n")


# In[7]:


# droping class column so as to consider features for pca
data.drop([0],axis=1,inplace=True)
data.head()


# In[8]:


# applying pca
x=data.values
pca=PCA(n_components = 0.95) # select number of components by preserving 95% of total variance
pca.fit(x)
x_pca=pca.transform(x)
x_pca=x_pca.round(2)


# In[9]:


print("# Percentage of Variance Explained by each of the selected components =\n",(pca.explained_variance_ratio_ *100))
print("\n")


# In[10]:


print("# Variance Explained by all the Principal Components =\n",np.sum(pca.explained_variance_ratio_ *100))
print("\n")


# In[11]:


print("# Cumulative sum of the Variance Explained by selected components =\n",np.cumsum(pca.explained_variance_ratio_ *100))
print("\n")


# In[12]:


# plotting Cumulative explained variance vs number of components
plt.plot(np.cumsum(pca.explained_variance_ratio_ *100))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance in %")
plt.savefig("fig_1.png",dpi=100)


# In[13]:


# ploting explained variance vs number of components
plt.plot(pca.explained_variance_ratio_ *100)
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance in %")
plt.savefig("fig_2.png",dpi=100)


# In[14]:


# ploting the fraph Eigenvalues vs number of components
ax = figure().gca()
ax.plot(pca.explained_variance_)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Number of Components')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, linewidth=1, color='r', alpha=0.5)
plt.title('Scree Plot of PCA: Component Eigenvalues')
plt.savefig("fig_3.png",dpi=100)
show()


# In[15]:


print("# Variance explained by first principal component     = ",np.cumsum(pca.explained_variance_ratio_ *100)[0])
print("# Variance explained by first 2 principal components  = ",np.cumsum(pca.explained_variance_ratio_ *100)[1])
print("# Variance explained by first 5 principal components  = ",np.cumsum(pca.explained_variance_ratio_ *100)[4])
print("# Variance explained by first 10 principal components = ",np.cumsum(pca.explained_variance_ratio_ *100)[9])
print("# Variance explained by first 15 principal components = ",np.cumsum(pca.explained_variance_ratio_ *100)[14])
print("# Variance explained by first 19 principal components = ",np.cumsum(pca.explained_variance_ratio_ *100)[18])
print("# Variance explained by first 20 principal components = ",np.cumsum(pca.explained_variance_ratio_ *100)[19])
print("\n")


# In[16]:


# df is dataframe having the features after running pca algorithm
df = pd.DataFrame(x_pca)
df=df.round(2)
df.head(35)


# In[17]:


# creating a new dataframe that containg features after running pca and also the column showing class of the data element
data_new = pd.DataFrame(x_pca)
data_new.head()
data_new['CLASS']=class_list
first_column = data_new.pop('CLASS')
data_new.insert(0, 'CLASS', first_column)
data_new=data_new.round(2)
data_new.head()


# In[18]:


md= df.to_markdown(index=False, tablefmt='pipe', colalign=['center']*len(df.columns))


# In[19]:


pca_md = open("/mnt/1CC48D25C48D026E/Computer Science/Sem 5/ML/assignment2_q1/pca_md.md","w")
pca_md.write(md)
pca_md.close()


# In[20]:


def euclidean_distance(point1, point2):
#return math.sqrt((point1[1]-point2[1])**2 + (point1[2]-point2[2])**2 + (point1[3]-point2[3])**2)   #sqrt((x1-x2)^2 + (y1-y2)^2)
# note that the following loop starts from 1 and not form 0 since the point1[0] contains the class of the data element which we are not considering in calculating distance
            val=0
            for i in range(1,len(point1)):
                val=val+((point1[i]-point2[i])**2)
            return math.sqrt(val)


# In[21]:


def kmean(data, k, max_iterations=500):
        # centroids is a dictionary that has the k centroids which are initially by starting data elements
        centroids = {}
        for i in range(k):
            centroids[i] = data.iloc[i].tolist()
        
        # we run following loop for max_iterations or till the time when there is no change in positions of centroids
        for i in range(max_iterations):
            # clusters is a dictionary that contains the k clusters
            clusters = {}
            # initialising the k clusters with empty list
            for j in range(k):
                clusters[j] = []
             # in following we are deciding to which one of the k clusters a data element belong to, we check this by comparing its distance
            # with all the centroids
            for point in range(data.shape[0]):
                distances = []
                for index in centroids.values():
                    distances.append(euclidean_distance(data.iloc[point].tolist(),index))
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(data.iloc[point].tolist())
            # previous is a dictionary that stores centroids of last iteration 
            previous = centroids.copy()
            # in following loop we are updating the centroids 
            for cluster_index in range(k):
                centroids[cluster_index] = np.average(clusters[cluster_index], axis = 0)
            # in following loop we are checking if the centriods of last iteration and current iteration are same or not                                                
            isOptimal=True
            for centroid in centroids:
                original_centroid = previous[centroid]
                curr = centroids[centroid]
                if (curr != original_centroid).all():
                    isOptimal = False
                
            if isOptimal:
                break
        return clusters


# In[22]:


# this is function to count the number of occurences of n in a list at it's list[i][0] th position
def count(l,n):
    cnt=0
    for i in range(len(l)):
        if(l[i][0]==n):
            cnt=cnt+1
    return cnt


# In[23]:


# function to calculate normalised mutual information 
def nmi(clusters, entropy_classes, k):
    entropy_clusters = 0            # to calculate entropy of cluster labels
    for i in range(len(clusters)):
        entropy_clusters = entropy_clusters + (((-1*(len(clusters[i])))/data.shape[0])*(np.log(((1*(len(clusters[i])))/data.shape[0]))))
    sum = 0
    
    for i in range(k):                                        
        # u is the number of data elements in i th cluster having 1 type of cancer
        u = (count(clusters[i],1))/len(clusters[i])
        # if no data element is present in cluster with type 1 cancer set u = 1 this helps us in further calculation (log)
        if(u==0):
            u=1
        v = (count(clusters[i],2))/len(clusters[i])
        if(v==0):
            v=1
        w = (count(clusters[i],3))/len(clusters[i])
        if(w==0):
            w=1
        sum = sum + (((-1)/k)*(((u)*(np.log(u)))+((v)*(np.log(v)))+((w)*(np.log(w)))))
    
    return (2*sum)/(entropy_classes+entropy_clusters) # returning the NMI
                                     
                       


# In[24]:


# scores stores (nmi) and k_val stores integers from 2 to 8
scores=[]
k_val=[]
# calling kmeans for k = 2 to 8
max_ = -111
for i in range(2,9):
    data = data_new
    c=kmean(data, i)
    val = nmi(c, entropy_classes, i)
    if(val>max_):
        max_=val
        ind=i
    scores.append(val)
    k_val.append(i)
print("\n")
print("K-MEANS CLUSTERING\n")
print("# Normalised Mutual Information for k in [2 ,8] = \n",scores)
print("\n")
print("# Maximum NMI is = ",max_, " and corresponding k is = ", ind)
print("\n")
    


# In[25]:


# plotting k vs NMI
fig, ax = plt.subplots()
ax.plot(k_val, scores, linewidth=2.0)
ax.set(xlim=(0, 8), xticks=np.arange(1, 10),
       ylim=(0, 1), yticks=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Normalised Mutual Information (NMI)")
plt.savefig("fig_4.png",dpi=100)
plt.show()


# In[26]:


# following code runs the kmeans once again on ind (k which gives max NMI) to show the clusters and the data elements it has
data = data_new
c=kmean(data, ind)
val = nmi(c, entropy_classes, ind)
print("# Following shows the clusters and the data elements present in it for k=",ind, " and in this case the NMI is = ", val)
print("\n")
for i in range(len(c)):
    print("Cluster ", i+1)
    print(c[i])
    print("\n")
print("\n")


# In[ ]:




