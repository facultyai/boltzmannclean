sherlockml-boltzmannclean
=========================

Fill missing values in a pandas DataFrame using a Restricted Boltzmann Machine.

Provides a class implementing the scikit-learn transformer interface for creating and training a Restricted Boltzmann Machine. This can then be sampled from to fill in missing values in training data or new data of the same format. Utility functions for applying the transformations to a pandas DataFrame are provided, with the option to treat columns as either continuous numerical or categorical features.

Installation
------------

.. code-block:: bash

    pip install sherlockml-boltzmannclean


Usage
-----

To fill in missing values from a DataFrame with the minimum of fuss, a cleaning function is provided.

.. code-block:: python

    import boltzmannclean

    my_clean_dataframe = boltzmannclean.clean(
        dataframe=my_dataframe,
        numerical_columns=['Height', 'Weight'],
        categorical_columns=['Colour', 'Shape'],
        tune_rbm=True  # tune RBM hyperparameters for my data
    )

To create and use the underlying scikit-learn transformer.

.. code-block:: python

    my_rbm = boltzmannclean.RestrictedBoltzmannMachine(
        n_hidden=100, learn_rate=0.01,
        batchsize=10, dropout_fraction=0.5, max_epochs=1,
        adagrad=True
    )

    my_rbm.fit_transform(a_numpy_array)

Here the default RBM hyperparameters are those listed above, and the numpy array operated on is expected to be composed entirely of numbers in the range [0,1] or np.nan/None. The hyperparameters are:

- *n_hidden*: the size of the hidden layer
- *learn_rate*: learning rate for stochastic gradient descent
- *batchsize*: batchsize for stochastic gradient descent
- *dropout_fraction*: fraction of hidden nodes to be dropped out on each backward pass during training
- *max_epochs*: maximum number of passes over the training data
- *adagrad*: whether to use the Adagrad update rules for stochastic gradient descent

Example
-------

.. code-block:: python

    import boltzmannclean
    import numpy as np
    import pandas as pd
    from sklearn import datasets

    iris = datasets.load_iris()

    df_iris = pd.DataFrame(iris.data,columns=iris.feature_names)
    df_iris['target'] = pd.Series(iris.target, dtype=str)

    df_iris.head()

=   =================   ================    =================   ================    ======
_   sepal length (cm)   sepal width (cm)    petal length (cm)   petal width (cm)    target
=   =================   ================    =================   ================    ======
0   5.1                  3.5                  1.4                  0.2                  0
1   4.9                  3.0                  1.4                  0.2                  0
2   4.7                  3.2                  1.3                  0.2                  0
3   4.6                  3.1                  1.5                  0.2                  0
4   5.0                  3.6                  1.4                  0.2                  0
=   =================   ================    =================   ================    ======


Add some noise:

.. code-block:: python

    noise = [(0,1),(2,0),(0,4)]

    for noisy in noise:
        df_iris.iloc[noisy] = None

    df_iris.head()

=   =================   ================    =================   ================    ======
_   sepal length (cm)   sepal width (cm)    petal length (cm)   petal width (cm)    target
=   =================   ================    =================   ================    ======
0   5.1                  NaN                  1.4                  0.2               None
1   4.9                  3.0                  1.4                  0.2                  0
2   NaN                  3.2                  1.3                  0.2                  0
3   4.6                  3.1                  1.5                  0.2                  0
4   5.0                  3.6                  1.4                  0.2                  0
=   =================   ================    =================   ================    ======

Clean the DataFrame:

.. code-block:: python

    df_iris_cleaned = boltzmannclean.clean(
        dataframe=df_iris,
        numerical_columns=[
            'sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)'
        ],
        categorical_columns=['target'],
        tune_rbm=True
    )

    df_iris_cleaned.round(1).head()


=   =================   ================    =================   ================    ======
_   sepal length (cm)   sepal width (cm)    petal length (cm)   petal width (cm)    target
=   =================   ================    =================   ================    ======
0   5.1                  3.3                  1.4                  0.2                  0
1   4.9                  3.0                  1.4                  0.2                  0
2   6.3                  3.2                  1.3                  0.2                  0
3   4.6                  3.1                  1.5                  0.2                  0
4   5.0                  3.6                  1.4                  0.2                  0
=   =================   ================    =================   ================    ======

The larger and more correlated the dataset is, the better the imputed values will be.