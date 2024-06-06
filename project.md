# Independent Project Work

Now that you have had a some time to work with the `scikit` workflow and see the differences in the main classes of problems we solve with machine learning, you will have the opportunity to work on a project of your own design. This is completely optional, but highly encouraged. You will not learn how to work with machine learning tools without practice and this is a great opportunity to get some practice. Below is a list of criteria that you should follow when designing your project. These will help you to get the most out of the experience.

## Classification Problem

A classification problem will use already tagged data (supervised learning) to predict the category of new observations. Here you might need to look for data or ask your research group if such data might exist for the problem you are working on. Here we recommend that you work with a data set that has at least 1000 observations or more. Training a model on a small data set can lead to overfitting and poor generalization. It would be good the number of features to be about 10 or more. This will give you a good sense of how to work with a data set that is not too small or too large. Having two many features would suggest that we instead use something like [dimension reduction](https://www.wikipedia.org/wiki/Dimensionality_reduction) to reduce the number of features before modeling -- this is good practice because big data sets can be computationally expensive to work with.

To successfully complete a classification problem, you should be able to:
* Read data into a pandas DataFrame
* Clean data and impute data (as needed) - we didn't do much imputation in this course, but it is a common practice in machine learning
* Plot data and propose potential models for analysis
* Split data into training and testing sets
* Build classification models
* Evaluate the quality of fitted models (here you might use a confusion matrix, ROC curve, or other metrics)

## Regression Problem

A regression problem will use data to predict a continuous outcome. Again, you might need to look for data or ask your research group if there's some data that might exist for you.