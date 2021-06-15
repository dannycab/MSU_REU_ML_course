#!/usr/bin/env python
# coding: utf-8

# # What is Tuning and Validation?
# 
# ## 1. What have done so far?
# 
# Thus far we have talked about two kinds of supervised machine learning problems: regression and classification. Fairly broad definitions of these two are below:
# 
# * **Regression**: Using a set of input data to predict an outcome on some continuous scale.
# * **Classification**: Using a set of input data to predict the class associated with data.
# 
# We have used `sci-kit learn` to begin to investigate how we can model data to solve one or the other problems. We have not talked in detail as to how these models work. That is important to understand if you are going to use particular models, but beyond the scope of this short course.
# 
# ## 2. Tuning and Validation
# 
# Instead, we will talk about the last pieces of supervised machine learning that are needed to understand your model: tuning and validation. Broad definitions are given below:
# 
# * **Tuning**: The process of finding the right model and hyperparamters to build an accurate model.
# * **Validation**: The process by which you build confidence in your model.
# 
# We will make use of `sci-kit learn`'s built in tools for tuning and validation. We will introduce those tools in class and we will focus on classifiers. For now, there are several useful videos to conceptually understand what we are trying to do.

# ## 3. Example with a Classifier
# 
# We will focus on classifiers because the process of tuning and validating them is a bit easier to understand at first. As we have seen we start our work with a classifier as follows:
# 
# 1. Read in the data
# 2. Clean/Transform data
# 3. Select model and parameters
# 4. Fit model
# 5. Evaluate model with confusion matrix
# 
# The message we want to convey is that parts 3, 4, and 5 often are part of a cyclkic process to adjust and change your model slightly to get a better prediction. *In fact, part 2 can come back also if you have to clean, encode, or impute your data differently.*
# 
# Because all the work we are doing relies on understanding the Confusion Matrix we will start there.

# ### 3.1 The Confusion Matrix
# 
# Watch the video below.

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo("Kdsp6soqA7o",width=640,height=360)


# ### 3.2 ROC
# 
# We can extract additional information about the quality of our model by varying the prediction threshold. That is, we allow the model to change the probability cutoff between predicting a positive (1) and negative (0) case. These resulting Receiver Operator Curve (ROC) can offer additional evidence as to the quality of your model beyond accuracy. In the video below, ROCs are described.

# In[2]:


from IPython.display import YouTubeVideo
YouTubeVideo("4jRBRDbJemM",width=640,height=360)


# ### 3.3 Leveraging randomness
# 
# As you might recall, we performed a data splitting when we started our modeling. That split was randomly done. So the Accuracy, ROC, and AUC were all deteermined for a single test set. What if we ran the model again? With a new random split? Would the results be similar our different? By how much?
# 
# You can see that there's a problem with running a single model and making a claim aboout it. Because our data was randomly split, our model produces results based on that split. If that split is representative of all possible splits then maybe it is ok to trust it. But if not it is better to build a bunch of models based on a bunch of random splits. Then you will get a disrtibution of results. That can give you some confidence in the predictions the model makes with a statistical uncertainty.
# 
# The video below talks about cross validation as one form of this. We will introduce this form and the [Monte Carlo](https://towardsdatascience.com/cross-validation-k-fold-vs-monte-carlo-e54df2fc179b) form.
# 
# Watch the video below.

# In[3]:


from IPython.display import YouTubeVideo
YouTubeVideo("fSytzGwwBVw",width=640,height=360)


# In[ ]:




