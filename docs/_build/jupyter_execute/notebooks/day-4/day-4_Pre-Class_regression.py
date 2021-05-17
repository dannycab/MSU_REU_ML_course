#!/usr/bin/env python
# coding: utf-8

# # Day-4 Pre-Class Assignment: Polynomial Regression
# 
# <img src="https://i.pinimg.com/originals/52/2c/20/522c209c019fe9592857bcb569184478.jpg">

# ## Goals for Pre-Class Assignment
# 
# After this pre-class assignment, you will be able to:
# 1. Generate data for a polynomial regression
# 2. Construct a set of polnomial regression models usings `statsmodels` 
# 3. Evaluate the quality of fit for a set of models using adjusted $R^2$ and determine the best fit
# 4. Explain why that model is the best fit for this data

# ## Our Imports

# In[1]:


import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML


# ---
# ## 1. Polynomial Regression
# 
# It's possible that a straight line is not going to be good enough to model the data we are working with. We can augment our $ Ax + B$ with extra features. By adding features we are still doing linear regression, but we the features themselves can consist of, well anything.
# 
# However, to be focused, for this pre-class we will use polynomials. We can add values like $x^2$ or $x^5$ to the potential set of features that can be used to better map against our data. 
# 
# <font size=8 color="#009600">&#9998;</font> Do This -  The question is, how many such features should we add? What are the advantages and disadvantages of adding more and more features? Think about it and answer in the cell below

# <font size=8 color="#009600">&#10174;</font> Answer here

# ### 1.1 Let's make some Data
# 
# It's always good when we are starting out to generate our own data. Data we generate gives us the advantage of **knowing** what the answer should be. 
# 
# <font size=8 color="#009600">&#9998;</font> Do This -  Let's do the following:
# * build a numpy array `x_ary`of values from -4 to 4 by 0.2
# * generate a corresponding `y_ary`, using the values from `x_ary`, based on the formula $x^4 + 2x^3 -15x^2 -12x + 36$
# * create `y_noisy`, by adding random (uniform) noise to `y_ary` in the range of -15 to 15. Later on we might make the range bigger (say -25 to 25) or smaller (say -5 to 5) for comparison.

# In[2]:


# your code here


# ### 1.2 Plot the data
# 
# We should really look at our data before we try to model it.
# 
# <font size=8 color="#009600">&#9998;</font> Do This -  plot `x_ary` vs both `y_ary` and `y_noisy`. Do it overlapping with colors, or side by side, whatever you think would look good. _Make sure to label your axes!_

# In[3]:


# your code here


# ---
# ## 2 Making the Polynomial Features
# 
# Ultimately it would be nice to do our work using a `pandas` DataFrame so that we have the opportunity to label our columns. There's the added benefit that `statsmodels` just works with `pandas` DataFrames. 
# 
# <font size=8 color="#009600">&#9998;</font> Do This - Make a DataFrame consisting of the following columns: a constant value for the intercept, the values in `x_ary`, and additional powers of `x_ary` up to 10.
# 
# You can do this one of two ways:
# 1. make the DataFrame out of `x_ary` and add features to the DataFrame
# 2. add columns to the `x_ary` array and then finish off by adding to a DataFrame
# 
# In the end, you have a DataFrame no matter the approach.
# 
# As a reminder, the columns of the DataFrame should be:
# * Label the first column "cnst" and just place the value 1 in it
# * make the `x_ary` data column 1, labeled "data"
# * the next 9 columns should be based on `x_ary` and have as values: $x^2$, $x^3$, $x^4 \ldots$ $x^{10}$. Give them good (but short) label names
# 
# Print the head of your DataFrame when you're done

# In[4]:


# your code


# ### 2.1 Fitting using the Polynomials
# 
# We'll talk about measures of "goodness" of fit during the class, but one good measure for a multi-feature fit is the **Adjusted R-squared** value. In general, the **R-squared** describes the variance in the model that it can account for. If the R-squared is 1.0, then all the variance is accounted for an you have a perfect fit. If the value is 0 and you have no fit. However, for multiple features R-squared tends to over-estimate. The Adjusted R-squared tries to deal with this and provide a value that is better suited to multiple features.
# 
# We'll leave it to you how you want to do this, but what we'd like you to try is to fit different combinations of features against `y_noisy` and report the Adjusted R-squared value. For example, what is the Adj-r-squared for:
# 
# 1. just the cnst column
# 2. the cnst and data column (which should be a line)
# 3. the cnst, data and $x^2$ column
# 4. the cnst, data, $x^2$ and $x^3$ column
# 5. $\ldots$
# 
# So on and so forth. You can do them individually or on a loop and collect the results. 
# 
# The value that is returned by the `.fit` method is an instance of a ` statsmodels v0.11.1 statsmodels.regression.linear_model.RegressionResults`. Run the `type` command on it and see. If you look on the <a href="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults"> statsmodels doc page </a> under "Properties" (look for that word as a title), you will find  all values you can gather from the variable returned by `.fit`. For this assignment the most important one of those is `.rsquared_adj`. 
# 
# <font size=8 color="#009600">&#9998;</font> Do This - Create a variety of models that fit to the noisy data using increasingly more features. Look at that value for the combination of features you selected and say which one is the "best". For this assignment, we would consider the "best" would be the highest value of `.rsquared_adj`.

# In[5]:


# your code here


# <font size=8 color="#009600">&#9998;</font> Do This - Which combination of features best "fit" your data? What was the Adjusted R-squared? Why might that combination produce the best fit?

# <font size=8 color="#009600">&#9998;</font> Answer here

# ---
# ## 3 Plot your data and your model
# 
# <font size=8 color="#009600">&#9998;</font> Do This -  Plot `x_ary` vs `y_noisy` and `x_ary` vs the best fitted values based on the adjusted rsquared value. Do it in the same graph. Again, the Property `.fittedvalues` gives out a panda Series with the fitted values. Also print out the summary for the variable returned by `.fit`

# In[6]:


# your code here


# ### 3.1 Are we justified in using this model?
# 
# As we did last class, we can check how well we are justified in using this model, by looking at the residual plot. 
# 
# <font size=8 color="#009600">&#9998;</font> Do This - Again, using `plot_regress_exog`, plot the residuals as a function of the independent variable (`data` or `x`, whatever you called it).

# In[7]:


## your code here


# <font size=8 color="#009600">&#9998;</font> Answer here - Do we appear justified in using this model? Why or why not? 
