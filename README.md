# Decision-Tree-Classifier
Decision Tree Classifier in Python with Scikit Learn

Decision Tree Algorithm Pseudocode

    1. Place the best attribute of our dataset at the root of the tree.
    2. Split the training set into subsets. Subsets should be made in such a way that each subset contains data with the same value      for an attribute.
    3. Repeat step 1 and step 2 on each subset until you find leaf nodes in all the branches of the tree.
    
# SkLearn Packages
      Python’s sklearn library holds tons of modules that help to build predictive models. It contains tools for data splitting, pre-processing, feature selection, tuning and supervised – unsupervised learning algorithms, etc. It is similar to Caret library in R programming.

For using it, we first need to install it. The best way to install data science libraries and its dependencies is by installing Anaconda package. You can also install only the most popular machine learning Python libraries.

Sklearn library provides us direct access to a different module for training our model with different machine learning algorithms like K-nearest neighbor classifier, Support vector machine classifier, decision tree, linear regression, etc.

# Decision Tree Training

Now we fit Decision tree algorithm on training data, predicting labels for validation dataset and printing the accuracy of the model using various parameters.

DecisionTreeClassifier(): This is the classifier function for DecisionTree. It is the main function for implementing the algorithms. Some important parameters are:

    criterion: It defines the function to measure the quality of a split. Sklearn supports “gini” criteria for Gini Index & “entropy” for Information Gain. By default, it takes “gini” value.
    splitter: It defines the strategy to choose the split at each node. Supports “best” value to choose the best split & “random” to choose the best random split. By default, it takes “best” value.
    max_features: It defines the no. of features to consider when looking for the best split. We can input integer, float, string & None value.
        If an integer is inputted then it considers that value as max features at each split.
        If float value is taken then it shows the percentage of features at each split.
        If “auto” or “sqrt” is taken then max_features=sqrt(n_features).
        If “log2” is taken then max_features= log2(n_features).
        If None, then max_features=n_features. By default, it takes “None” value.
    max_depth: The max_depth parameter denotes maximum depth of the tree. It can take any integer value or None. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. By default, it takes “None” value.
    min_samples_split: This tells above the minimum no. of samples reqd. to split an internal node. If an integer value is taken then consider min_samples_split as the minimum no. If float, then it shows percentage. By default, it takes “2” value.
    min_samples_leaf: The minimum number of samples required to be at a leaf node. If an integer value is taken then consider min_samples_leaf as the minimum no. If float, then it shows percentage. By default, it takes “1” value.
    max_leaf_nodes: It defines the maximum number of possible leaf nodes. If None then it takes an unlimited number of leaf nodes. By default, it takes “None” value.
    min_impurity_split: It defines the threshold for early stopping tree growth. A node will split if its impurity is above the threshold otherwise it is a leaf.

Let’s build classifiers using criterion as gini index & information gain. We need to fit our classifier using fit(). We will plot our decision tree classifier’s visualization too.

# Confusion Matrix:
      In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix.
A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.
It allows easy identification of confusion between classes e.g. one class is commonly mislabeled as the other. Most performance measures are computed from the confusion matrix.

