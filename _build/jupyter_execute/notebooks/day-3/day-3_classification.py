#!/usr/bin/env python
# coding: utf-8

# # Classification Model
# 
# <img src="https://www.pngkey.com/png/full/574-5745035_machine-learning-workflow-machine-learning-data-pipeline.png" width=700px>
# 
# 
# ## Agenda for today's class
# 
# </p>
# 
# 1. Review pre-class assignment
# 1. Train vs Test
# 1. Example classifier using breast cancer dataset
# 1. logistic regression classifier

# ## 0. Imports for the day

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics


# ---
# ## 1. Review of Pre-Class assignment
# 
# We'll discussion any questions that came up as a class.

# ----
# ## 2. Training vs Testing
# 
# As you learned in the pre-class, classification is an ML process that maps features of an input data set to class labels. Classification is a **supervised** learning approach where example data is used to train the data. We typically divide the data used to train and evaluate the classifier (the result model) into three sets
# 
# - training set
# - testing set
# - validation set

# &#9989; **Do This:** As a group, discuss what these three sets represent. It might help to review these terms on the web. The answers down below:

# <font size=8 color="#009600">&#9998;</font> Training set is:

# <font size=8 color="#009600">&#9998;</font> Testing set is:

# <font size=8 color="#009600">&#9998;</font> Validation set is:

# **Defining the features and building the model**
# 
# If you review the image at the top of the notebook, you might notice that one of the first steps in machine learning is to go from "raw data" into a set of "features" and "labels", which we discussed a bit about when we worked with the perceptron model. Extracting features from our data can sometimes be one of the trickier parts of the process and also one of the most important ones. We have to think carefully about exactly what the "right" features are for training our machine learning algorithm and, when possible, it is advantageous to find ways to reduce the total number of features we are trying to model. Once we define our features, we can build our model.

# &#9989; **Do This:** Now, also as a group, discuss and be prepared to share with the class how you would define a feature vector from an image and how you would build a face recognition model using machine learning.  Can you find any libraries that you think would be will suited for this? Which of these libraries might you actually use?

# <font size=8 color="#009600">&#9998;</font> Do This - Erase the contents of this cell and put a basic outline of what your group discussed.

# ### 2.1 Working with data
# 
# There is a common data set used to work with classification called the breast cancer data set. It is actually available in `sklearn` but what fun is working withe cleaned up data. Let's look at the original. In the directory where this notebook is stored is "breast-cancer-wisconsin.data" and "breast-cancer-wisconsin.names". The data are in ".data" and the ".names" describes that data. 
# 
# Read in the data, label the columns based on the .names file. Look at the dtypes, anything unusual? Why?

# <font size=8 color="#009600">&#9998;</font> What's unusual about dtypes? Why?

# In[2]:


# your code here


# Can you write code to identify what the problem is? That is, can you provide a DataFrame of the offending rows that are causing the problem? There are lots of ways to do this and, frankly, it is probably a bit hard so don't get hung up too long on this. Give it a try though:

# In[3]:


# your code here


# OK, we have an imputation problem. Write code to solve it and say what you did.
# 
# By the way, there is an argument `na_values` that you can provide to `read_csv` that will mark a list of characters as if they were `np.nan` using `na_values`, which is pretty darn convenient. Using that will help when importing the data for classification. 

# In[4]:


# code here


# ----
# ### 2.2 : Splitting the dataset for model into training and testing sets
# Let's split the data in a training set and final testing set. We want to randomly select 75% of the data for training and 25% of the data for testing.
# 
# You should turn the `class_labels` into 0 (now 2, for benign) and 1 (now 4, for malignant) as the classifier we are using (Logisitic Regression) predicts valuse between 0 and 1.

# &#9989; Do This - You will need to come up with a way to split the data into separate training and testing sets (we will leave the validation set out for now).  Make sure you keep the feature vectors and classes together.  
# 
# **BIG HINT**: This is a very common step in machine learning, andthere exists a function to do this for you in the `sklearn` library called `train_test_split`. From the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), you find that takes the features and class labels as input and returns a 4 outputs:
# - 2 feature sets (one for training and one for testing)
# - 2 class labels sets (the corresponding one for training and for testing)
# 
# Use `train_test_split` to split your data into a training set and a testing set that correspond to 75% and 25% of your data respectively. Check the length of the resulting output to make sure the splits follow what you expected.

# **Question**: Why do we need to separate our samples into a training and testing set. Why can't we just use all the data for both? Wouldn't that make it work better?

# <font size=8 color="#009600">&#9998;</font> Do This - Erase the contents of this cell and replace it with your answer to the above question!  (double-click on this text to edit this cell, and hit shift+enter to save the text)

# ----
# ## 3 Logistic Regression
# 
# In the pre-class, you watched a video explaining some of the aspects of the logistic regression. The full details on logistic regression require deeper study, but we can gain some insight from looking at the function we are trying to fit to our data. Plot out the curve to the following equation, called the **logistic function***
# 
# $$ f(x) = \frac{e^{x}} {1+e^{x} }  \equiv \frac{1}{1+e^{-x} } $$
# 
# Plot that from say -6 to 6.

# In[5]:


# your code


# What is interesting about that curve is that all values of x are mapped into the range for y of 0.0-1.0. Assuming you have a binary classifier, that is one that only has two class labels, that is the mapping you want: all combination of features map into the two class labels 0, or 1. Moreover, the graph looks a lot like a [cumulative probability distribution](https://en.wikipedia.org/wiki/Cumulative_distribution_function). The probability that a set of features is of class 1 is 1 to the right and 0 to the left. As you watched in the pre-class video, this is the basis for logistic regression.
# 
# It's a regression because the "x" in our logisitic function is actually going to be a regression equation, such that
# 
# $$ fn= b_{0} + b_{1}x_{1} + b_{2}x_{2} + \ldots  $$
# 
# for as many terms as we like and the new logistic function
# 
# $$ f(x) = \frac{e^{fn(x))}} {1+e^{fn(x)} }  \equiv \frac{1}{1+e^{-fn(x)} } $$
# 
# Logisitic classification tries to find the parameters $b_{x}$ that gives maximal performance on training, and hopefully testing. Let's let `statsmodels` do that.
# 
# We are going to use all the training data from above and train a logistic regression. It is similar to what we did before with regular regression, to wit:
# 
# Note, very importantly, the `sm.add_constant` on the train vectors. We talked about that when we did OLS in stat models. That column of constant is what the $b_{0}$ or intercept will train against. We need that column to get an intercept.

# In[6]:


## Uncomment to run
# logit_model = sm.Logit(train_labels, sm.add_constant(train_vectors))
# result = logit_model.fit()
# print(result.summary() )


# The "Pseudo R-squ" is the equivalent (mostly) of the R-squared value in Linear regression. Ranges from 0 (poor fit) to 1 (perfect fit). The P values under "P > |z|" are measures of significance. The null hypothesis is that the restricted model (say a constant value for `fn`) performs better but a low p-value suggests that we can reject this hypothesis and prefer the full model over the null model. This is similar to the F-test for linear regression.
# 
# Based on that, remove the low-performing columns, reform the train test sets and run it again, print the summary

# In[7]:


# your code here


# **Question:** How do the fits of the full model and the reduced model compare? What evidence are you using to make compare these two fits?

# <font size=8 color="#009600">&#9998;</font> Answer here.

# ### 3.1 How'd it go?
# 
# There are a number of ways that we can check the performance of our model; we learn new ways throughout the semester. The major difference in the standard statistics approach and supervised learning approaches is that we test our models using the data that we held out: "the test data." 
# 
# That is, we will use our classifier model to make predictions from the test features and we can then compare those predictions to actual test labels. To test accuracy, we can use the output of the `.fit()` method of the model to predict how well the classifier works on the test data (the data it was not trained on). Conveniently that is the `.predict()` method and, again, we use it on the result of the `.fit()`. 
# 
# **Note:** The output from `.predict()` is not a 0/1 value as the test labels are, but rather a fraction between 0 and 1 indicating how likely each entry is to be one class or another. We can make the assumption that anything greater than 0.5 would be a 1 class and anything less than 0.5 would be a 0 class. 
# 
# <font size=8 color="#009600">&#9998;</font> So do the following:
# - use the `.predict()` method (look it up) to create the predicted labels using the test input
# - convert the output of the `.predict()` method to the 0/1 class values of the test labels
# - print the resulting predicted class values of the test labels

# In[8]:


# your code here


# One of the first metrics we will use is the `accuracy_score`, which compares the predictions our model made for the test labels and the actual test labels. The `accuracy_score` is one of many metrics we can use and is included in `sklearn.metrics`. Here's the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score) on `accuracy_score`.
# 
# <font size=8 color="#009600">&#9998;</font> Do this:
# - Use the `sklearn.metrics` we imported at the top and run the `accuracy_score` on the 0/1 predicted label and the test labels.
# - Print your accuracy result

# In[9]:


## your code here


# **Question:** How well did your model predict the test class labels? Given what you learned in the pre-class assignment about false positives and false negatives, what other questions should we ask about the accuracy of our model?
