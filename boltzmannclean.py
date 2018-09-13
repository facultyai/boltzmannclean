from __future__ import division, print_function

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.model_selection import GridSearchCV

COLNAME_SEPARATOR = "_boltzmannclean"


def clean(dataframe, numerical_columns, categorical_columns, tune_rbm):
    """Replaces missing values in dataframe with imputed values from an RBM"""

    numerics, scaler = preprocess_numerics(dataframe, numerical_columns)

    categoricals, category_dict = preprocess_categoricals(
        dataframe, categorical_columns
    )

    preprocessed_array = np.hstack((numerics, categoricals))

    if preprocessed_array.size > 0:
        # create and train a Restricted Boltzmann Machine
        rbm = train_rbm(preprocessed_array, tune_hyperparameters=tune_rbm)

        imputed_array = rbm.transform(preprocessed_array)

        imputed_numerics = postprocess_numerics(
            imputed_array[:, : numerics.shape[1]], dataframe[numerical_columns], scaler
        )

        imputed_categoricals = postprocess_categoricals(
            imputed_array[:, numerics.shape[1] :],
            dataframe[categorical_columns],
            category_dict,
        )

        imputed_dataframe = imputed_numerics.join(imputed_categoricals)

        for colname in imputed_dataframe.columns:
            dataframe[colname] = imputed_dataframe[colname]

    return dataframe


def preprocess_categoricals(dataframe, categorical_columns):
    """
    Encode categorical dataframe columns for a Restricted Boltzmann Machine.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A Pandas DataFrame to be used for training an RBM on.
    categorical_columns : list of str
        A list of the column names to be treated as categorical values.

    Returns
    -------
    encoded : np.array
        A numpy array of the categorical columns encoded for RBM training.
    category_dict: dict
        A dictionary mapping categorical column names to a list of categories.

    """

    categoricals = pd.DataFrame(dataframe[categorical_columns])

    if not categoricals.empty:
        # label encoding
        category_dict = {}
        for colname in categoricals.columns:
            category_dict[colname] = (
                categoricals[colname].astype("category").cat.categories
            )
            categoricals[colname] = categoricals[colname].astype("category").cat.codes

        categoricals[categoricals == -1] = np.nan

        # one-hot encoding
        encoded = pd.get_dummies(
            categoricals.astype("object"), prefix_sep=COLNAME_SEPARATOR
        )

        col_iterator = encoded.items()

        # reinstate our nulls
        for colname in categoricals.columns:
            nulls = categoricals.index[categoricals[colname].isnull()]
            for _ in range(len(category_dict[colname])):
                encoded_colname, __ = next(col_iterator)
                encoded.loc[nulls, encoded_colname] = np.nan

        # extract the numpy array
        encoded = encoded.values
    else:
        encoded = np.empty((dataframe.shape[0], 0))
        category_dict = {}

    return encoded, category_dict


def reverse_dummy_encoding(dummy_dataframe, category_dict):
    """
    Reconstructs the original from a one-hot encoded dataframe.

    """

    current_col = 0
    col_positions = {}

    for colname, values in category_dict.items():
        col_positions[colname] = list(range(current_col, current_col + len(values)))
        current_col += len(values)

    dataframe = pd.DataFrame(
        {
            colname: pd.Categorical.from_codes(
                np.argmax(
                    dummy_dataframe.iloc[:, col_positions[colname]].values, axis=1
                ),
                category_dict[colname],
            )
            for colname in category_dict.keys()
        },
        dtype="object",
    )

    return dataframe


def postprocess_categoricals(imputed_array, original_dataframe, category_dict):
    """
    Recreate a categorical dataframe from values imputed by an RBM.

    Parameters
    ----------
    imputed_array : np.array
        A numpy array with values in the range [0,1].
    original_dataframe : pd.DataFrame
        The DataFrame with the columns used to create the original numpy array.
    category_dict: dict
        A dictionary mapping categorical column names to a list of categories.

    Returns
    -------
    dataframe : pd.DataFrame
        A DataFrame with columns reconstructed from the input array.

    """

    dataframe = pd.DataFrame(imputed_array)

    if not dataframe.empty:
        dataframe = reverse_dummy_encoding(dataframe, category_dict)

        dataframe.index = original_dataframe.index

        for colname in dataframe.columns:
            dataframe[colname] = dataframe[colname].astype(
                original_dataframe[colname].dtype
            )

    return dataframe


def preprocess_numerics(dataframe, numerical_columns):
    """
    Preprocess numerical dataframe columns for a Restricted Boltzmann Machine.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A Pandas DataFrame to be used for training an RBM on.
    numerical_columns : list of str
        A list of the column names to be treated as numerical values.

    Returns
    -------
    numerics : np.array
        A numpy array of the numerical columns scaled to [0,1].
    scaler: sklearn.preprocessing.MinMaxScaler
        The scikit-learn scaler used to transform the values.

    """

    # converts to numerical values where possible, replaces with NaN if not
    numerics = pd.DataFrame(dataframe[numerical_columns]._convert(numeric=True))
    # selects only columns with some numerical values
    numerics = numerics.select_dtypes([np.number])

    if not numerics.empty:
        to_impute = np.logical_not(np.isfinite(numerics))

        # avoids that annoying pandas warning
        numerics.is_copy = False
        # replaces infs with nans
        numerics[to_impute] = np.nan

        # replace NaNs with column means to leave min-max scaling unaffected
        array = Imputer().fit_transform(numerics)

        # scale values to the range [0,1]
        scaler = MinMaxScaler().fit(array)
        numerics = scaler.transform(array)

        # put our NaNs back in to be imputed by the RBM
        numerics[to_impute] = np.nan
    else:
        numerics = np.empty((dataframe.shape[0], 0))
        scaler = None

    return numerics, scaler


def postprocess_numerics(imputed_array, original_dataframe, scaler):
    """
    Recreate a numerical dataframe from values imputed by an RBM.

    Parameters
    ----------
    imputed_array : np.array
        A numpy array with values in the range [0,1].
    original_dataframe : pd.DataFrame
        The DataFrame with the columns used to create the original numpy array.
    scaler: sklearn.preprocessing.MinMaxScaler
        The scikit-learn scaler used to transform the values.

    Returns
    -------
    dataframe : pd.DataFrame
        A DataFrame with numerical columns reconstructed from the input array.

    """
    if scaler:
        array = scaler.inverse_transform(imputed_array)
    else:
        array = np.empty_like(imputed_array)

    imputed_dataframe = pd.DataFrame(array, index=original_dataframe.index)
    imputed_dataframe.columns = (
        original_dataframe._convert(numeric=True).select_dtypes([np.number]).columns
    )

    for colname in imputed_dataframe.columns:
        imputed_dataframe[colname] = imputed_dataframe[colname].astype(
            original_dataframe[colname].dtype
        )

    return imputed_dataframe


def train_rbm(array, tune_hyperparameters):
    """
    Creates and trains a Restricted Boltzmann Machine.

    Parameters
    ----------
    array : np.array
        A numpy array of np.nan's and floats in the range [0,1].
    tune_hyperparameters : bool
        Tune learn rate and hidden layer size of the RBM, train for longer.

    Returns
    -------
    rbm : RestrictedBoltzmannMachine
        A trained RBM which implements the scikit-learn transformer interface

    """

    rbm = RestrictedBoltzmannMachine(
        n_hidden=array.shape[1], batchsize=min(array.shape[0], 10)
    )

    if tune_hyperparameters:
        rbm.max_epochs = 50
        param_grid = {
            "n_hidden": [rbm.n_hidden // 10, rbm.n_hidden, rbm.n_hidden * 10],
            "adagrad": [True, False],
        }
        grid_search = GridSearchCV(rbm, param_grid=param_grid)
        grid_search.fit(array)
        rbm.verbose = True

    rbm.fit(array)

    return rbm


class RestrictedBoltzmannMachine(BaseEstimator, TransformerMixin):
    """
    Restricted Boltzmann Machine trained by Persistent Contrastive Divergence.

    Creates and trains a Restricted Boltzmann Machine which can then be used to
    fill in missing values in a numpy array. Trained using Persistent
    Contrastive Divergence (PCD) gradient descent (optionally with adagrad).
    Implements scikit-learn's transformer interface, uses training techniques
    and notation from Geoffrey Hinton.

    See "A Practical Guide to Training Restricted Boltzmann Machines"
    (https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) for more info.

    Parameters
    ----------
    n_hidden : int, optional
        Size of the hidden layer for the RBM.
    learn_rate : float, optional
        Initial learning rate for gradient descent.
    batchsize : int, optional
        The size of a gradient descent minibatch.
    dropout_fraction : float, optional
        Fraction of hidden units to drop out in the PCD phase of PCD.
    max_epochs : int, optional
        Maximum number of training epochs allowed.
    adagrad : bool, optional
        Whether to use the adagrad algorithm during gradient descent.
    verbose : bool, optional
        Whether to print out epoch numbers during fitting.

    """

    def __init__(
        self,
        n_hidden=100,
        learn_rate=0.01,
        batchsize=10,
        dropout_fraction=0.5,
        max_epochs=1,
        adagrad=True,
        verbose=False,
    ):
        self.n_hidden = n_hidden
        self.learn_rate = learn_rate
        self.batchsize = batchsize
        self.dropout_fraction = dropout_fraction
        self.max_epochs = max_epochs
        self.adagrad = adagrad
        self.verbose = verbose

    def score_samples(self, X):

        noise_fraction = 0.5

        mask = np.random.random(size=X.shape)

        noisy_X = X.copy()
        noisy_X[mask < noise_fraction] = np.nan

        X_reco = self.transform(noisy_X)

        scores = -1 * np.sum(np.square(X_reco - np.nan_to_num(X)), axis=1)

        return scores

    def score(self, X, y=None):
        score = np.sum(self.score_samples(X))
        return score

    def transform(self, X, y=None):

        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))

        X_reco = self._reconstruct_until_stable(
            np.nan_to_num(X), self.w_, self.a_, self.b_, fixed_values=np.isfinite(X)
        )

        return X_reco

    def fit(self, X, y=None):

        if self.verbose:
            print("Training RBM...")

        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))

        X = X[np.isfinite(X).any(axis=1)]

        n_visible = X.shape[1]
        num_examples = X.shape[0]
        n_hidden = self.n_hidden
        batchsize = self.batchsize

        w = 0.01 * np.random.randn(n_visible, n_hidden)
        delta_w = np.zeros_like(w)

        prior = np.true_divide(np.nan_to_num(X).sum(axis=0), num_examples)
        prior[prior == 1] = 0.9999

        a = np.log(1 / (1 - prior))
        a = np.reshape(a, (1, n_visible))
        delta_a = np.zeros_like(a)

        b = -4 * np.ones((1, n_hidden))
        delta_b = np.zeros_like(b)

        weight_decay = 0.01

        if self.adagrad:
            w_adagrad = np.ones_like(w)
            a_adagrad = np.ones_like(a)
            b_adagrad = np.ones_like(b)
            adagrad_eps = 1e-8

        n_batches = num_examples // batchsize

        pcd_chain = np.random.random_sample((batchsize, n_visible))

        for epoch in range(self.max_epochs):
            if self.verbose:
                print("\rEpoch ", epoch + 1, " of ", self.max_epochs, end="")

            np.random.shuffle(X)

            for batch in range(n_batches):

                # positive phase of contrastive divergence

                v0 = X[int(batch * batchsize) : int((batch + 1) * batchsize)]

                # deal with nans
                visible_dropout = 1 - np.isfinite(v0).sum(axis=1) / v0.shape[1]

                nan_scaling = 1.0 / (1 - visible_dropout)
                nan_scaling = nan_scaling.reshape((v0.shape[0], 1))

                v0 = np.nan_to_num(v0)

                prob_h0, h0 = self._sample_hidden(v0, w, b)

                # negative phase of (persistent) contrastive divergence

                pcd_chain = self._reconstruct(
                    pcd_chain, w, a, b, dropout_fraction=self.dropout_fraction
                )

                prob_h, h = self._sample_hidden(
                    pcd_chain, w, b, dropout_fraction=self.dropout_fraction
                )

                # gradient for weights
                vh0 = np.dot((v0 * nan_scaling).T, prob_h0)
                vh = np.dot(pcd_chain.T, prob_h)

                # gradient for visible biases
                positive_visible_grad = np.sum(v0, axis=0)
                negative_visible_grad = np.sum(pcd_chain, axis=0)

                # gradient for hidden biases
                positive_hidden_grad = np.sum(prob_h0, axis=0)
                negative_hidden_grad = np.sum(prob_h, axis=0)

                m = 0.5 if epoch < 5 else 0.9

                w_grad = vh0 - vh
                a_grad = positive_visible_grad - negative_visible_grad
                b_grad = positive_hidden_grad - negative_hidden_grad

                if self.adagrad:
                    w_adagrad += np.square(w_grad)
                    a_adagrad += np.square(a_grad)
                    b_adagrad += np.square(b_grad)

                    w_learn_rate = self.learn_rate / np.sqrt(adagrad_eps + w_adagrad)

                    a_learn_rate = self.learn_rate / np.sqrt(adagrad_eps + a_adagrad)

                    b_learn_rate = self.learn_rate / np.sqrt(adagrad_eps + b_adagrad)
                else:
                    w_learn_rate = self.learn_rate
                    a_learn_rate = self.learn_rate
                    b_learn_rate = self.learn_rate

                delta_w = delta_w * m + (w_learn_rate / batchsize) * (w_grad)
                delta_a = delta_a * m + (a_learn_rate / batchsize) * (a_grad)
                delta_b = delta_b * m + (b_learn_rate / batchsize) * (b_grad)

                #  L1 weight decay
                w += delta_w - weight_decay * np.sign(w) * (self.learn_rate / batchsize)
                a += delta_a
                b += delta_b

        self.w_ = w
        self.a_ = a
        self.b_ = b
        if self.verbose:
            print("\n")

        return self

    def _sample_hidden(self, v, w, b, dropout_fraction=0.0):
        prob_h = self._logistic(v, w, b)
        h = prob_h > np.random.rand(v.shape[0], self.n_hidden)
        h = np.multiply(
            h,
            np.random.binomial(
                [np.ones((v.shape[0], self.n_hidden))], 1 - dropout_fraction
            )[0]
            * (1.0 / (1 - dropout_fraction)),
        )
        return prob_h, h

    def _logistic(self, x, w, b):
        xw = np.dot(x, w)
        return 1 / (1 + np.exp(-xw - b))

    def _free_energy(self, v, w, a, b):
        vw = np.dot(v, w)
        return -1 * (np.dot(v, a.T) + np.sum(np.log(1 + np.exp(b + vw))))

    def _reconstruct(self, v0, w, a, b, dropout_fraction=0.0):
        _, h = self._sample_hidden(v0, w, b, dropout_fraction)
        v = self._logistic(h, w.T, a)
        return v

    def _reconstruct_until_stable(self, v, w, a, b, fixed_values=[], threshold=0.01):

        reco_free_energy = self._free_energy(v, w, a, b)
        reco_free_energy_new = reco_free_energy.copy()

        batchsize = v.shape[0]

        delta_free_energy = np.ones((batchsize, 1))
        converged = np.zeros((batchsize,), dtype=bool)

        v_true = v.copy()

        if self.verbose:
            print("Reconstructing", batchsize, "examples...")

        while converged.sum() < batchsize:
            unconverged = np.logical_not(converged)

            v[unconverged] = self._reconstruct(v[unconverged], w, a, b)

            v[fixed_values] = v_true[fixed_values]

            reco_free_energy_new[unconverged] = self._free_energy(
                v[unconverged], w, a, b
            )

            delta_free_energy[unconverged] = np.abs(
                reco_free_energy_new[unconverged] - reco_free_energy[unconverged]
            ) / (reco_free_energy[unconverged] + 1e-8)

            converged = (delta_free_energy < threshold).flatten()

            reco_free_energy = reco_free_energy_new

        return v
