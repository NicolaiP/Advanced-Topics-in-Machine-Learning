"""
Nicolai F. Pedersen, nicped@dtu.dk
This code is an adaption of a python module for regularized kernel canonical correlation analysis, which can be found on
https://github.com/gallantlab/pyrcca
"""

import sys

import h5py
import joblib
import numpy as np
from scipy.linalg import eigh
from sklearn.cross_decomposition.pls_ import _center_scale_xy

sys.path.append("../")  # go to parent dir
sys.path.append("../..")  # go to parent dir


class _CCABase(object):
    def __init__(self, reg=None, n_components=None, cutoff=1e-15):
        self.reg = reg
        self.n_components = n_components
        self.cutoff = cutoff

    def fit(self, X, Y):
        # print('Training CCA, regularization = %0.4f, %d components' % (self.reg, self.n_components))

        X = X.copy()
        Y = Y.copy()

        # Subtract mean and divide by std
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (_center_scale_xy(X, Y))
        data = [X, Y]

        # Get dimensions of data and number of canonical components
        kernel = [d.T for d in data]
        nDs = len(kernel)
        nFs = [k.shape[0] for k in kernel]
        numCC = min([k.shape[1] for k in kernel]) if self.n_components is None else self.n_components

        # Get the auto- and cross-covariance matrices
        crosscovs = [np.dot(ki, kj.T) / len(ki.T - 1) for ki in kernel for kj in kernel]

        # Allocate left-hand side (LH) and right-hand side (RH):
        LH = np.zeros((sum(nFs), sum(nFs)))
        RH = np.zeros((sum(nFs), sum(nFs)))

        # Fill the left and right sides of the eigenvalue problem
        # Eq. (7) in https://www.frontiersin.org/articles/10.3389/fninf.2016.00049/full
        for ii in range(nDs):
            RH[sum(nFs[:ii]):sum(nFs[:ii + 1]), sum(nFs[:ii]):sum(nFs[:ii + 1])] = \
                (crosscovs[ii * (nDs + 1)] + self.reg[ii] * np.eye(nFs[ii]))

            for jj in range(nDs):
                if ii != jj:
                    LH[sum(nFs[:jj]): sum(nFs[:jj + 1]), sum(nFs[:ii]): sum(nFs[:ii + 1])] = crosscovs[nDs * jj + ii]

        # The matrices are symmetric, i.e. A = A^T, this makes sure that small differences are evened out.
        LH = (LH + LH.T) / 2.
        RH = (RH + RH.T) / 2.

        maxCC = LH.shape[0]

        # Solve the generalized eigenvalue problem for the two symmetric matrices
        # Returns the eigenvalues and the eigenvectors
        r, Vs = eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
        r[np.isnan(r)] = 0
        rindex = np.argsort(r)[::-1]
        comp = []
        Vs = Vs[:, rindex]
        for ii in range(nDs):
            comp.append(Vs[sum(nFs[:ii]):sum(nFs[:ii + 1]), :numCC])

        self.x_weights_ = comp[0]
        self.y_weights_ = comp[1]

        self.x_loadings_ = np.dot(self.x_weights_.T, crosscovs[0]).T
        self.y_loadings_ = np.dot(self.y_weights_.T, crosscovs[-1]).T

        return self

    def transform(self, X, Y):
        check_is_fitted(self, 'x_mean_')
        check_is_fitted(self, 'y_mean_')
        X = X.copy()
        Y = Y.copy()
        X -= self.x_mean_
        X /= self.x_std_
        Y -= self.y_mean_
        Y /= self.y_std_
        x_scores = np.dot(X, self.x_weights_)
        y_scores = np.dot(Y, self.y_weights_)
        return x_scores, y_scores

    def transform2(self, X, Y):
        check_is_fitted(self, 'x_mean_')
        check_is_fitted(self, 'y_mean_')
        X = X.copy()
        Y = Y.copy()
        X -= self.x_mean_
        X /= self.x_std_
        Y -= self.y_mean_
        Y /= self.y_std_
        x_scores = np.dot(self.x_weights_.T, X.T)
        y_scores = np.dot(self.y_weights_.T, Y.T)
        return x_scores, y_scores

    def train(self, data):
        print('Training CCA, regularization = %0.4f, %d components' % (self.reg, self.n_components))

        comps = kcca(data, self.reg, self.n_components)
        self.cancorrs, self.ws, self.comps = recon(data, comps)
        if len(data) == 2:
            self.cancorrs = self.cancorrs[np.nonzero(self.cancorrs)]
        return self

    def validate(self, vdata):
        vdata = [np.nan_to_num(_zscore(d)) for d in vdata]
        if not hasattr(self, 'x_weights_'):
            raise NameError('Algorithm has not been trained.')
        self.preds, self.corrs = predict(vdata, [self.x_weights_, self.y_weights_], self.cutoff)
        return self.corrs

    def compute_ev(self, vdata):
        """
        This function estimates the variance explained (R^2) in the test data by each of the canonical components.

        :param vdata:
        :return:
        """
        nD = len(vdata)
        nC = self.ws[0].shape[1]
        nF = [d.shape[1] for d in vdata]
        self.ev = [np.zeros((nC, f)) for f in nF]
        for cc in range(nC):
            ccs = cc + 1
            print('Computing explained variance for component #%d' % ccs)
            preds, corrs = predict(vdata, [w[:, ccs - 1:ccs] for w in self.ws], self.cutoff)
            resids = [abs(d[0] - d[1]) for d in zip(vdata, preds)]
            for s in range(nD):
                ev = abs(vdata[s].var(0) - resids[s].var(0)) / vdata[s].var(0)
                ev[np.isnan(ev)] = 0.
                self.ev[s][cc] = ev
        return self.ev

    def save(self, filename):
        h5 = h5py.File(filename, 'a')
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, list):
                    for di in range(len(value)):
                        grpname = 'dataset%d' % di
                        dgrp = h5.require_group(grpname)
                        try:
                            dgrp.create_dataset(key, data=value[di])
                        except RuntimeError:
                            del h5[grpname][key]
                            dgrp.create_dataset(key, data=value[di])
                else:
                    h5.attrs[key] = value
        h5.close()

    def load(self, filename):
        h5 = h5py.File(filename, 'a')
        for key, value in h5.attrs.items():
            setattr(self, key, value)
        for di in range(len(h5.keys())):
            ds = 'dataset%d' % di
            for key, value in h5[ds].items():
                if di == 0:
                    setattr(self, key, [])
                self.__getattribute__(key).append(value.value)


class CCACrossValidate(_CCABase):
    """
    Attributes:
        numCV (int): number of cross-validation folds
        regs (list or numpy.array): regularization param array.
                                   Default: np.logspace(-3, 1, 10)
        numCCs (list or numpy.array): list of numbers of canonical dimensions
                                     to keep. Default is np.range(5, 10).
    Returns:
        ws (list): canonical weights
        comps (list): canonical components
        cancorrs (list): correlations of the canonical components
                         on the training dataset
        corrs (list): correlations on the validation dataset
        preds (list): predictions on the validation dataset
        ev (list): explained variance for each canonical dimension
    """

    def __init__(self, numCV=None, regs=None, numCCs=None, select=0.2, cutoff=1e-15):
        self.numCCs = np.arange(5, 10) if numCCs is None else numCCs
        self.select = select
        self.regs = np.array(np.logspace(-3, 1, 10)) if regs is None else regs
        self.numCV = 10 if numCV is None else numCV
        super(CCACrossValidate, self).__init__(cutoff=cutoff)

    def train(self, data, parallel=True):
        """
        Train CCA with cross-validation for a set of regularization
        coefficients and/or numbers of CCs
        Attributes:
            data (list): training data matrices
                         (number of samples X number of features).
                         Number of samples must match across datasets.
            parallel (bool): use joblib to train cross-validation folds
                             in parallel
        """
        corr_mat = np.zeros((len(self.regs), len(self.numCCs)))
        selection = max(int(self.select * min([d.shape[1] for d in data])), 1)
        for ri, reg in enumerate(self.regs):
            for ci, numCC in enumerate(self.numCCs):
                running_corr_mean_sum = 0.

                # Run in parallel
                if parallel:
                    fold_corr_means = joblib.Parallel(n_jobs=self.numCV)(joblib.delayed(train_cvfold)
                                                                         (data=data, reg=reg, numCC=numCC,
                                                                          cutoff=self.cutoff, selection=selection
                                                                          ) for fold in range(self.numCV))
                    running_corr_mean_sum += sum(fold_corr_means)

                # Run in sequential
                else:
                    for cvfold in range(self.numCV):
                        fold_corr_mean = train_cvfold(data=data, reg=reg, numCC=numCC, cutoff=self.cutoff, selection=selection)
                        running_corr_mean_sum += fold_corr_mean

                corr_mat[ri, ci] = running_corr_mean_sum / self.numCV
        best_ri, best_ci = np.where(corr_mat == corr_mat.max())
        self.best_reg = self.regs[best_ri[0]]
        self.best_numCC = self.numCCs[best_ci[0]]

        comps = kcca(data, self.best_reg, self.best_numCC)

        self.cancorrs, self.ws, self.comps = recon(data, comps)
        if len(data) == 2:
            self.cancorrs = self.cancorrs[np.nonzero(self.cancorrs)]
        return self


def train_cvfold(data, reg, numCC, cutoff, selection):
    """
    Train a cross-validation fold of CCA
    """
    nT = data[0].shape[0]
    chunklen = 10 if nT > 50 else 1
    nchunks = int(0.2 * nT / chunklen)
    indchunks = list(zip(*[iter(range(nT))] * chunklen))
    np.random.shuffle(indchunks)
    heldinds = [ind for chunk in indchunks[:nchunks] for ind in chunk]
    notheldinds = list(set(range(nT)) - set(heldinds))
    comps = kcca([d[notheldinds] for d in data], reg, numCC)
    cancorrs, ws, ccomps = recon([d[notheldinds] for d in data], comps)
    preds, corrs = predict([d[heldinds] for d in data], ws, cutoff=cutoff)
    fold_corr_mean = []
    for corr in corrs:
        corr_idx = np.argsort(corr)[::-1]
        corr_mean = corr[corr_idx][:selection].mean()
        fold_corr_mean.append(corr_mean)
    return np.mean(fold_corr_mean)


class CCA(_CCABase):
    """Attributes:
        reg (float): regularization parameter. Default is 0.1.
        n_components (int): number of canonical dimensions to keep. Default is 10.
        kernelcca (bool): kernel or non-kernel CCA. Default is True.
        ktype (string): type of kernel used if kernelcca is True.
                        Value can be 'linear' (default) or 'gaussian'.
        verbose (bool): default is True.
    Returns:
        ws (list): canonical weights
        comps (list): canonical components
        cancorrs (list): correlations of the canonical components
                         on the training dataset
        corrs (list): correlations on the validation dataset
        preds (list): predictions on the validation dataset
        ev (list): explained variance for each canonical dimension
    """

    def __init__(self, reg1=0.2, reg2=0.2, n_components=10, cutoff=1e-15):
        reg = [reg1, reg2]
        super(CCA, self).__init__(reg=reg, n_components=n_components, cutoff=cutoff)

    def train(self, data):
        # # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (_center_scale_xy(data[0], data[1]))
        return super(CCA, self).train([X, Y])


def predict(vdata, ws, cutoff=1e-15):
    """Get predictions for each dataset based on the other datasets
    and weights. Find correlations with actual dataset."""
    iws = [np.linalg.pinv(w.T, rcond=cutoff) for w in ws]
    ccomp = _listdot([d.T for d in vdata], ws)
    ccomp = np.array(ccomp)
    preds = []
    corrs = []

    for dnum in range(len(vdata)):
        idx = np.ones((len(vdata),))
        idx[dnum] = False
        proj = ccomp[idx > 0].mean(0)
        pred = np.dot(iws[dnum], proj.T).T
        pred = np.nan_to_num(_zscore(pred))
        preds.append(pred)
        cs = np.nan_to_num(_rowcorr(vdata[dnum].T, pred.T))
        corrs.append(cs)
    return preds, corrs


def kcca(data, reg=0., numCC=None):
    """Set up and solve the kernel CCA eigenproblem
    """
    kernel = [d.T for d in data]
    nDs = len(kernel)
    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[1] for k in kernel]) if numCC is None else numCC

    # Get the auto- and cross-covariance matrices
    crosscovs = [np.dot(ki, kj.T) / len(ki.T - 1) for ki in kernel for kj in kernel]

    # Allocate left-hand side (LH) and right-hand side (RH):
    LH = np.zeros((sum(nFs), sum(nFs)))
    RH = np.zeros((sum(nFs), sum(nFs)))

    # Fill the left and right sides of the eigenvalue problem
    # Eq. (7) in https://www.frontiersin.org/articles/10.3389/fninf.2016.00049/full
    for ii in range(nDs):
        RH[sum(nFs[:ii]):sum(nFs[:ii + 1]), sum(nFs[:ii]):sum(nFs[:ii + 1])] = \
            (crosscovs[ii * (nDs + 1)] + reg * np.eye(nFs[ii]))

        for jj in range(nDs):
            if ii != jj:
                LH[sum(nFs[:jj]): sum(nFs[:jj + 1]), sum(nFs[:ii]): sum(nFs[:ii + 1])] = crosscovs[nDs * jj + ii]

    LH = (LH + LH.T) / 2.
    RH = (RH + RH.T) / 2.

    maxCC = LH.shape[0]

    # Solve the generalized eigenvalue problem for the two symmetric matrices
    r, Vs = eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
    r[np.isnan(r)] = 0
    rindex = np.argsort(r)[::-1]
    comp = []
    Vs = Vs[:, rindex]
    for ii in range(nDs):
        comp.append(Vs[sum(nFs[:ii]):sum(nFs[:ii + 1]), :numCC])

    return comp


def recon(data, comp, corronly=False):
    # Get canonical variates and CCs
    ws = comp
    ccomp = _listdot([d.T for d in data], ws)
    corrs = _listcorr(ccomp)
    if corronly:
        return corrs
    else:
        return corrs, ws, ccomp


def _zscore(d): return (d - d.mean(0)) / d.std(0)


def _demean(d): return d - d.mean(0)


def _listdot(d1, d2): return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]


def _listcorr(a):
    """Returns pairwise row correlations for all items in array as a list of matrices
    """
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j > i:
                corrs[:, i, j] = [np.nan_to_num(np.corrcoef(ai, aj)[0, 1]) for (ai, aj) in zip(a[i].T, a[j].T)]
    return corrs


def _rowcorr(a, b):
    """Correlations between corresponding matrix rows"""
    cs = np.zeros((a.shape[0]))
    for idx in range(a.shape[0]):
        cs[idx] = np.corrcoef(a[idx], b[idx])[0, 1]
    return cs


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


if __name__ == '__main__':

    pass
