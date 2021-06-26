# Authors: Andreas Mueller <andreas.mueller@columbia.edu>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
# License: BSD 3 clause

import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, _safe_indexing
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion, Pipeline

import pdb
__all__ = ['TransformedTargetRegressor']


class TransformedTargetRegressor(RegressorMixin, BaseEstimator):
    """Meta-estimator to regress on a transformed target.

    Useful for applying a non-linear transformation to the target ``y`` in
    regression problems. This transformation can be given as a Transformer
    such as the QuantileTransformer or as a function and its inverse such as
    ``log`` and ``exp``.

    The computation during ``fit`` is::

        regressor.fit(X, func(y))

    or::

        regressor.fit(X, transformer.transform(y))

    The computation during ``predict`` is::

        inverse_func(regressor.predict(X))

    or::

        transformer.inverse_transform(regressor.predict(X))

    Read more in the :ref:`User Guide <transformed_target_regressor>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    regressor : object, default=None
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
        If regressor is ``None``, ``LinearRegression()`` is created and used.

    transformer : object, default=None
        Estimator object such as derived from ``TransformerMixin``. Cannot be
        set at the same time as ``func`` and ``inverse_func``. If
        ``transformer`` is ``None`` as well as ``func`` and ``inverse_func``,
        the transformer will be an identity transformer. Note that the
        transformer will be cloned during fitting. Also, the transformer is
        restricting ``y`` to be a numpy array.

    func : function, default=None
        Function to apply to ``y`` before passing to ``fit``. Cannot be set at
        the same time as ``transformer``. The function needs to return a
        2-dimensional array. If ``func`` is ``None``, the function used will be
        the identity function.

    inverse_func : function, default=None
        Function to apply to the prediction of the regressor. Cannot be set at
        the same time as ``transformer`` as well. The function needs to return
        a 2-dimensional array. The inverse function is used to return
        predictions to the same space of the original training labels.

    check_inverse : bool, default=True
        Whether to check that ``transform`` followed by ``inverse_transform``
        or ``func`` followed by ``inverse_func`` leads to the original targets.

    Attributes
    ----------
    regressor_ : object
        Fitted regressor.

    transformer_ : object
        Transformer used in ``fit`` and ``predict``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.compose import TransformedTargetRegressor
    >>> tt = TransformedTargetRegressor(regressor=LinearRegression(),
    ...                                 func=np.log, inverse_func=np.exp)
    >>> X = np.arange(4).reshape(-1, 1)
    >>> y = np.exp(2 * X).ravel()
    >>> tt.fit(X, y)
    TransformedTargetRegressor(...)
    >>> tt.score(X, y)
    1.0
    >>> tt.regressor_.coef_
    array([2.])

    Notes
    -----
    Internally, the target ``y`` is always converted into a 2-dimensional array
    to be used by scikit-learn transformers. At the time of prediction, the
    output will be reshaped to a have the same number of dimensions as ``y``.

    See :ref:`examples/compose/plot_transformed_target.py
    <sphx_glr_auto_examples_compose_plot_transformed_target.py>`.

    """
    @_deprecate_positional_args
    def __init__(self, regressor=None, *, transformer=None,
                 func=None, inverse_func=None, check_inverse=True,
                 check_array_X=False, check_array_y=False):
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.check_array_X = check_array_X
        self.check_array_y = check_array_y

    def _fit_transformer(self, y):
        """Check transformer and fit transformer.

        Create the default transformer, fit it and make additional inverse
        check on a subset (optional).

        """
        if (self.transformer is not None and
                (self.func is not None or self.inverse_func is not None)):
            raise ValueError("'transformer' and functions 'func'/"
                             "'inverse_func' cannot both be set.")
        elif self.transformer is not None:
            self.transformer_ = clone(self.transformer)
        else:
            if self.func is not None and self.inverse_func is None:
                raise ValueError("When 'func' is provided, 'inverse_func' must"
                                 " also be provided")
            self.transformer_ = FunctionTransformer(
                func=self.func, inverse_func=self.inverse_func, validate=True,
                check_inverse=self.check_inverse)
        # XXX: sample_weight is not currently passed to the
        # transformer. However, if transformer starts using sample_weight, the
        # code should be modified accordingly. At the time to consider the
        # sample_prop feature, it is also a good use case to be considered.
        self.transformer_.fit(y)
        if self.check_inverse:
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = _safe_indexing(y, idx_selected)
            y_sel_t = self.transformer_.transform(y_sel)
            if not np.allclose(y_sel,
                               self.transformer_.inverse_transform(y_sel_t)):
                warnings.warn("The provided functions or transformer are"
                              " not strictly inverse of each other. If"
                              " you are sure you want to proceed regardless"
                              ", set 'check_inverse=False'", UserWarning)

    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            Parameters passed to the ``fit`` method of the underlying
            regressor.


        Returns
        -------
        self : object
        """
        if self.check_array_y:
            y = check_array(y, accept_sparse=False, force_all_finite=True,
                            ensure_2d=False, dtype='numeric')

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            # import pdb; pdb.set_trace()
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        if self.regressor is None:
            from sklearn.linear_model import LinearRegression
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        self.regressor_.fit(X, y_trans, **fit_params)

        return self

    def predict(self, X):
        """Predict using the base regressor, applying inverse.

        The regressor is used to predict and the ``inverse_func`` or
        ``inverse_transform`` is applied before returning the prediction.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            Predicted values.

        """
        check_is_fitted(self)
        pred = self.regressor_.predict(X)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if (self._training_dim == 1 and
                pred.ndim == 2 and pred.shape[1] == 1):
            pred = pred.squeeze(axis=1)

        return pred

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination :math:`R^2` of the
        prediction.
        The coefficient :math:`R^2` is defined as :math:`(1 - \\frac{u}{v})`,
        where :math:`u` is the residual sum of squares ``((y_true - y_pred)
        ** 2).sum()`` and :math:`v` is the total sum of squares ``((y_true -
        y_true.mean()) ** 2).sum()``. The best possible score is 1.0 and it
        can be negative (because the model can be arbitrarily worse). A
        constant model that always predicts the expected value of `y`,
        disregarding the input features, would get a :math:`R^2` score of
        0.0.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.
        Notes
        -----
        The :math:`R^2` score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """
        if self.check_array_y:
            y = check_array(y, accept_sparse=False, force_all_finite=True,
                            ensure_2d=False, dtype='numeric')

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            # import pdb; pdb.set_trace()
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y_trans, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}

    @property
    def n_features_in_(self):
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() returns False the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute."
                .format(self.__class__.__name__)
            ) from nfe

        return self.regressor_.n_features_in_