#!/usr/bin/env python
# coding: utf-8

# # Using PCA

# ## Goals
# 
# After completing this notebook, you will be able to:
# 1. Standardize data with `scikit-learn`
# 2. Perform Principal Component Analysis (PCA) on data
# 3. Evaluate the influence of different principal components by seeing how much variance they explain
# 4. Be able to transform data into lower dimensions uing PCA
# 5. Be able to use KernelPCA to separate nonlinearly separable data

# ## 0. Our Imports

# In[1]:


# Standard Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PCA Imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

# Import for 3d plotting
from mpl_toolkits import mplot3d

# For making nonlinear data
from sklearn.datasets import make_circles

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Getting Example Data
# 
# Today we'll be looking at sklearn's breast cancer identification dataset. You could get this directly from sklearn with `from sklearn.datasets import load_breast_cancer`, but its good to practice reading in data so we'll do it by hand. There are two files you'll need for this data:
# 
# - `cancer.csv` contains the cell measurements
# - `target.csv` has if each cell is malignant (1) or benign (0).
# 
# `https://raw.githubusercontent.com/dannycab/MSU_REU_ML_course/main/notebooks/day-5/cancer.csv`
# 
# `https://raw.githubusercontent.com/dannycab/MSU_REU_ML_course/main/notebooks/day-5/target.csv`
# 
# <font size=8 color="#009600">&#9998;</font> Do This - Read in these files as separate DataFrames with `pd.read_csv()`and print ther `head()`s.

# In[2]:


# your code here


# ### 1.1 Getting the right columns
# 
# Note that the dataframe from `cancer.csv` has a column that is just the index again. This isn't actually part of the data so you should get rid of it. 
# 
# 
# <font size=8 color="#009600">&#9998;</font> Use the DataFrame `drop()` method to make a new DataFrame that doesen't have this extra column.

# In[3]:


# your code here


# How many features does this data have? Does all of the data fall on the same scale?

# ### 1.2 Scaling the Data
# 
# 
# Like you saw in the previous notebook, to use PCA to reduce the number of features, the data needs to be scaled. 
# 
# 
# <font size=8 color="#009600">&#9998;</font> Do this - use `sklearn.preprocessing.StandardScaler` to scale the `cancer.csv` data. 

# In[4]:


# your code here


# ## 2. Appplying PCA
# 
# Now that the data is scaled appropriatley, we're ready to actually use Principal Component analysis. 
# 
# The syntax for doing PCA with scikit-learn is similar to other classes we've worked with from sklearn. You first initialize an instance of the PCA class (this is where you choose hyperparameters), then fit it to your data.  ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html))
# 
# 
# ### 2.1 Task
# <font size=8 color="#009600">&#9998;</font>  Create and fit a PCA object without specifying any hyperparameters.

# In[5]:


# your code here


# <font size=8 color="#009600">&#9998;</font> What is the total explained variance from all of the principal components? How many principal components are there? (you'll probably want to write some code to figure this out)

# In[6]:


# your code here


# answer here

# ### 2.2 Explained Variance Curve
# 
# It can be useful to visually see how much variance each principal component explains. 
# 
# <font size=8 color="#009600">&#9998;</font> Make a plot of the running total of the explained variance from each principal component. 

# In[7]:


# your code here


# <font size=8 color="#009600">&#9998;</font> About how many Principal Components does it take to cover $90\%$ of the variance in the data?

# answer here

# ### 2.3 Transforming Data with PCA
# 
# To move data into a lower dimensional space with PCA, we can use the `transform(data)` method. Since we can only visualize data in $2$ and $3$ dimensions, let's try transforming data using $2$ and $3$ principal components.
# 
# #### 2.3.1 2d Transform
# 
# <font size=8 color="#009600">&#9998;</font> Do this - Create a new pca object with only 2 principal components and transform the data into this 2d space. How much of the total variance is explained by these two principal components?
# 
# (note: Is there another method that lets you fit and transform with only one line of code?)
# 

# In[8]:


# your code here


# #### 2.3.2 2d Transformed Data Plot
# 
# <font size=8 color="#009600">&#9998;</font> Do this - Make a scatter plot of the transformed data (now in 2d) with the two principal components as axes. Color the points in the scatter plot by if the corresponding cell is malignant or not.
# 
# Hint: To get the colors working in the `c = ` argument, try casting the dataframe with the target data as a numpy array. You'll need to transpose / index the array to get the data in the form matplotlib needs.

# In[9]:


# your code here


# #### 2.3.3 Observations
# 
#  What kind of structure do you see in this transformed data? Is it seperable? If yes (or mostly yes), **how** is it separable?

# answer here

# #### 2.3.3 3d Transform
# 
# <font size=8 color="#009600">&#9998;</font> Repeat the procedure from 2.3.1 - 2.3.3 but in 3d. Here some [documentation](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html) on `mplot3d`, which you can use to make a 3d scatter plot.

# In[10]:


# your code here


# ## 3. (Time Permitting) Kernel PCA
# 
# Standard PCA is extremely useful for data that is linearly separable, but it falls short when data has nonlinear structure. In those cases, Kernel PCA may be of use. Kernel PCA is similar in concept to Support Vector Machines (SVMs), as it works by using a kernel function to move data into a higher dimensional space, where the data may become linearly seperable, and hence a typical PCA is effective.
# 
# Here's some circles like we saw last class:

# In[11]:


X, y = make_circles(n_samples=500, factor=0.3, noise=0.06, random_state=10)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.axis('equal')
plt.show()


# ### 3.1 Standard PCA on Nonlinear Data
# 
# <font size=8 color="#009600">&#9998;</font> Try using a regular PCA to transform this data (but don't change the dimensionality) and plot the results.

# In[12]:


# your code here


# <font size=8 color="#009600">&#9998;</font> What does your group observe?

# answer here

# ### 3.2 Trying Kernel PCA
# 
# <font size=8 color="#009600">&#9998;</font> Now try using a Kernel PCA to transform the data. Try to find a kernel and set of parameters that separate the data linearly. What do you observe? (Hint: the `'rbf'` kernel is sensitive to different values of `gamma`)

# In[13]:


# your code here


# answer here

# A Few Other Dimension Reduction Techniques of note:
# 
# - Linear Discriminant Analysis (LDA): Used with classifiers, finds the linear combination of features for separating the data.
# - t-distributed stochastic neighbor embedding (t-SNE): Nonlinear Method, good for visualizing datasets in 2 and 3 dimensions. 
#  - Incremental PCA: Like a standard PCA but can be much more memory efficient in some cases.
# 
# This list is very much non-exhaustive.
