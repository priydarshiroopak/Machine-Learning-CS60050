#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# In[2]:


# analysing the dataset
dataset = pd.read_csv("Data_C.csv")


# In[3]:


dataset.size


# In[4]:


dataset.shape


# In[5]:


dataset.head(5)


# In[6]:


# encoding the dicrete values
dataset["Gender"] = dataset["Gender"].replace(["Male","Female"],[1,0])
dataset["Vehicle_Age"]=dataset["Vehicle_Age"].replace(["> 2 Years","1-2 Year","< 1 Year"],[2,1,0])
dataset["Vehicle_Damage"]=dataset["Vehicle_Damage"].replace(["Yes","No"],[1,0])
dataset.head()                                                                 


# In[7]:


dataset.describe()


# In[8]:


dataset.isnull().sum()


# In[9]:


dataset.hist(bins=12, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0, 0, 1.2, 1.2)) 


# In[10]:


dataset.head(20)


# In[11]:


# features has the column names
features = dataset.keys()
features = features.drop('id')
features = features.drop('Response').tolist()


# In[ ]:





# In[12]:


# y is target and X has other columns than target
X = dataset.drop("Response", axis=1)
y = dataset["Response"]


# In[13]:


# Node class to store nodes of trees
class Node:
    def __init__(self, name=None):
        self.list_=[]
        self.type=""
        self.name=name
        self.entropy = None
        self.result=None
        self.split_point=None                             # splitting point for continuous feature
        self.continuous=0                                 # if the node feature is continuous then this =1
        if len(self.list_)==0:
            self.type='Leaf Node'
        else:
            self.type='Decision Node'
            


# In[14]:


# calculating entropy
def calculate_entropy(l):
    
    entropy = 0.
    if len(l) != 0:
        p1 = len(l[l==1]) / len(l) 
        if p1 != 0 and p1 != 1:
             entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        else:
             entropy = 0
          
    return entropy


# In[15]:


# function to calculate information gain of continous feature 
def information_gain_continuous(dataset_modified, feature_j, node_entropy):
    df = dataset_modified.filter([feature_j,"Response"],axis=1)
    df = df.sort_values(feature_j)
    information_gain_cont = 0
    for i in range(df.shape[0]):
        r = df["Response"]
        p = r[0:i+1]
        q = r[i+1: ]
        z = node_entropy - (((len(p)/len(r))*(calculate_entropy(p)))+((len(q)/len(r))*(calculate_entropy(q))))
        if (z>information_gain_cont):
            information_gain_cont = z
                   
    return information_gain_cont


# In[16]:


# function to calculate information gain of discrete function 
def calculate_information_gain(dataset_modified, feature_j, node_entropy):
    
    information_gain = 0
    weighted_entropy = 0
    feature_j_values = dataset_modified[feature_j].unique().tolist()
    for i in range(len(feature_j_values)):
        l = np.array(dataset_modified[dataset_modified[feature_j]==feature_j_values[i]]["Response"])
        entropy = (calculate_entropy(l))
        w = ((l.size)/(dataset_modified.shape[0]))
        weighted_entropy = weighted_entropy + (w* entropy)
    
    information_gain = node_entropy - weighted_entropy
    
    return information_gain


# In[17]:


# this function calculates the best feature for the next node
def best_split(dataset_modified, features, node_entropy):   
    
    max_information_gain=0
    best_feature = None    
    for i in range(len(features)):
        if(features[i]== 'Age'or features[i]== 'Region_Code'or features[i]=='Annual_Premium' or features[i]== 'Vintage'or features[i]== 'Policy_Sales_Channel'):
            information_gain = information_gain_continuous(dataset_modified, (features[i]), node_entropy)
        else:
            information_gain = calculate_information_gain(dataset_modified, (features[i]), node_entropy)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = features[i]
                
    return best_feature


# In[18]:


# this function finds the splitting point (point to split the data in two parts) for continuos feature
def find_split_point(dataset_modified, best_feature, node_entropy):
    split_point=None
    df = dataset_modified.filter([best_feature,"Response"],axis=1)
    df = df.sort_values(best_feature)
    information_gain_cont = 0
    for i in range(df.shape[0]):
        r = np.array(df["Response"])
        p = r[0:i+1]
        q = r[i+1: ]
        z = node_entropy - (((len(p)/len(r))*(calculate_entropy(p)))+(((len(q)/len(r)))*(calculate_entropy(q))))
        if (z>information_gain_cont):
            information_gain_cont = z
            split_point = ((2*i)+1)/2
            
    return split_point


# In[19]:


# this function builds all the nodes recursively starting from root node
def build_node(dataset_modified, features, depth_current, node_entropy,max_depth,tree, ind =None):

        best_feature= best_split(dataset_modified, features, node_entropy)
        max_depth = max(max_depth,depth_current)
        if(best_feature == None):
            return max_depth
        else:
            m = dataset_modified["Response"]
            node_entropy = calculate_entropy(m)

            node=Node(best_feature)
            if(best_feature== 'Age'or best_feature== 'Region_Code'or best_feature=='Annual_Premium' or best_feature== 'Vintage'or best_feature== 'Policy_Sales_Channel'):
                node.continuous = 1
                node.split_point = find_split_point(dataset_modified, best_feature, node_entropy)

            else:
                pass
            if(len(dataset[dataset["Response"]==1]) > len(dataset[dataset["Response"]==0])):
                node.result = 1
            else:
                node.result = 0
            node.entropy = node_entropy
            tree.append(node)
            if(depth_current==0):
                ind = 0
            else:
                (tree[ind].list_).append(node)
                ind = len(tree)-1
            if(best_feature== 'Age'or best_feature== 'Region_Code'or best_feature=='Annual_Premium' or best_feature== 'Vintage'or best_feature== 'Policy_Sales_Channel'):
                best_feature_values = ["greater than s","less than equal to s"]
            else:

                best_feature_values = np.sort((dataset_modified[best_feature].unique())).tolist()
            features.remove(best_feature)
            for j in range(len(best_feature_values)):
                dataset_modified = dataset_modified[dataset_modified[best_feature] == best_feature_values[j]]
                build_node (dataset_modified, features, depth_current + 1, node_entropy,max_depth,tree,ind)
        return max_depth


# In[20]:


# this function buids tree by calling build_node
def build_tree(dataset, features, depth_current, node_entropy,tree, ind =None):
    
    max_depth = 1    
    max_depth=build_node(dataset,features,0,node_entropy,max_depth,tree)
    return max_depth


# In[21]:


# node_entropy here is the entropy of root node, this is calculated since we are passing node_entropy to buil_tree function
p1 = len(y[y==1])/len(y)
node_entropy = (((-1)*p1)*(np.log2(p1)))-(((1-p1))*(np.log2(1-p1)))


# In[22]:


features


# In[23]:


# function to traverse the tree
def traverse_tree(testset,node,output):

    for i in range(testset.shape[0]):
        if not (len(node.list_)==0):
            if(node.name== 'Age'or node.name== 'Region_Code'or node.name=='Annual_Premium' or node.name== 'Vintage'or node.name== 'Policy_Sales_Channel'):
                if(testset.iloc[i].at[node.name] <= node.split_point):
                    return traverse_tree(testset,node.list_[0],output)
                elif(testset.iloc[i].at[node.name] > node.split_point):
                    return traverse_tree(testset,node.list_[len(node.list_)-1],output)
            else:
                if(testset.iloc[i].at[node.name]==0):
                    return traverse_tree(testset,node.list_[0],output)
                elif(testset.iloc[i].at[node.name]==1):
                    return traverse_tree(testset,node.list_[len(node.list_)-1],output)
        else:
            output.append(node.result)
    
    return

     


# In[24]:


# function to calculate the accuracy
def calculate_accuracy(output,expected):
    pos_count=0
    if not (len(output)==0):
        for i in range(len(output)):
            if(output[i]==expected[i]):
                pos_count+=1
        accuracy=(pos_count/len(output))*100
        return accuracy
    else:
        return 0


# In[25]:


# purning of tree
def prune_tree(node,node_entropy,threshold):
    
    if not len(node.list_)==0:
        n=node.entropy - node.list_[0].entropy
        m=node.entropy - node.list_[len(node.list_)-1].entropy
        if(n < threshold or m < threshold):
            node.list_=[]
        else:
            node_entropy = node.list_[0].entropy
            return prune_tree(node.list_[0],node_entropy,threshold)
            node_entropy = node.list_[len(node.list_)-1].entropy
            return prune_tree(node.list_[len(node.list_)-1],node_entropy,threshold)
    else:
        return 


# In[26]:


# function to print the tree
def print_tree(node,count):
    if not(len(node.list_)==0):
        for i in range(count):
            print("\t")
        print("Depth: ",count-1, node.name)
        if not len(node.list_)==0:
            print_tree(node.list_[0],count+1)
            print_tree(node.list_[len(node.list_)-1],count+1)
    else:
        print("Depth: ",count-1, node.name)
    


# In[27]:


# in this function we split the dataset in 80% train, 20% test part, ten times randomly 
# and then we print max_accuracy, and depth of the tree with max accuracy. And then we prune this tree and print it.
def train_test_split(dataset, features,node_entropy):
    
    randomlist = []
    treelist=[]
    accuracylist=[]
    maxdepth=[]
    tree=[]
    index = int(0.2 * (dataset.shape[0]))
    
    for i in range(10):
        n = random.randint(0, (dataset.shape[0]-index))
        randomlist.append(n)
        
    for i in range(10):
        
        output=[]
        dataset_train = dataset.drop(dataset.index[randomlist[i]:randomlist[i]+index])
        dataset_test = dataset.iloc[randomlist[i]:randomlist[i]+index,:]
        expected = (dataset_test["Response"]).tolist()
        dataset_test = dataset_test.drop("Response", axis=1)
        
        max_depth = build_tree(dataset_train, features, 0, node_entropy,tree)
        maxdepth.append(max_depth)
        treelist.append(tree)
        traverse_tree(dataset_test,tree[0],output)
        accuracy = calculate_accuracy(output, expected)
        accuracylist.append(accuracy)
        
    max_accuracy = max(accuracylist)  
    max_accuracy_ind=accuracylist.index(max(accuracylist))
    tree = treelist[max_accuracy_ind]
    prune_tree(tree[0],node_entropy,0.1)
    maxdepth_=max(maxdepth)
    print("accuracy: ",max_accuracy)
    print("depth of tree: ",maxdepth_)
    print_tree(tree[0],1)
    df = pd.DataFrame(output, columns=["output"])
    df.to_csv('output1.csv', index=False)
    return tree


# In[ ]:


# calling train_test_split
train_test_split(dataset,features,node_entropy)


# In[ ]:





# In[ ]:




