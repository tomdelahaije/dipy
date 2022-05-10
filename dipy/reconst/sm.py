"""Classes and functions for fitting the standard model of diffusion in neuronal
tissue."""

from warnings import warn

import numpy as np

from dipy.reconst.base import ReconstModel


def design_matrix(btens):
    """Calculate the design matrix from the b-tensors.

    Parameters
    ----------
    btens : numpy.ndarray
        An array of b-tensors of shape (number of acquisitions, 3, 3).

    Returns
    -------
    X : numpy.ndarray
        Design matrix.

    Notes
    -----
    The design matrix is generated according to

        .. math::

            X = \begin{pmatrix} 1 & -\mathbf{b}_1^T & \frac{1}{2}(\mathbf{b}_1
            \otimes\mathbf{b}_1)^T \\ \vdots & \vdots & \vdots \\ 1 &
            -\mathbf{b}_n^T & \frac{1}{2}(\mathbf{b}_n\otimes\mathbf{b}_n)^T
            \end{pmatrix}
    """
    X = np.zeros((btens.shape[0], 28))
    for i, bten in enumerate(btens):
        b = from_3x3_to_6x1(bten)
        b_sq = from_6x6_to_21x1(b @ b.T)
        X[i] = np.concatenate(
            ([1], (-b.T)[0, :], (0.5 * b_sq.T)[0, :]))
    return X


def _ols_fit(data, mask, X, step=int(1e4)):
    """Estimate the model parameters using ordinary least squares.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (..., number of acquisitions).
    mask : numpy.ndarray
        Boolean array with the same shape as the data array of a single
        acquisition.
    X : numpy.ndarray
        Design matrix of shape (number of acquisitions, 28).
    step : int, optional
        The number of voxels over which the fit is calculated simultaneously.

    Returns
    -------
    params : numpy.ndarray
        Array of shape (..., 28) containing the estimated model parameters.
        Element 0 is the natural logarithm of the estimated signal without
        diffusion-weighting, elements 1-6 are the estimated diffusion tensor
        elements in Voigt notation, and elements 7-27 are the estimated
        covariance tensor elements in Voigt notation.
    """
    params = np.zeros((np.product(mask.shape), 28)) * np.nan
    data_masked = data[mask]
    size = len(data_masked)
    X_inv = np.linalg.pinv(X.T @ X)  # Independent of data
    if step >= size:  # Fit over all data simultaneously
        S = np.log(data_masked)[..., np.newaxis]
        params_masked = (X_inv @ X.T @ S)[..., 0]
    else:  # Iterate over data
        params_masked = np.zeros((size, 28))
        for i in range(0, size, step):
            S = np.log(data_masked[i:i + step])[..., np.newaxis]
            params_masked[i:i + step] = (X_inv @ X.T @ S)[..., 0]
    params[np.where(mask.ravel())] = params_masked
    params = params.reshape((mask.shape + (28,)))
    return params


def _wls_fit(data, mask, X, step=int(1e4)):
    """Estimate the model parameters using weighted least squares with the
    signal magnitudes as weights.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (..., number of acquisitions).
    mask : numpy.ndarray
        Array with the same shape as the data array of a single acquisition.
    X : numpy.ndarray
        Design matrix of shape (number of acquisitions, 28).
    step : int, optional
        The number of voxels over which the fit is calculated simultaneously.

    Returns
    -------
    params : numpy.ndarray
        Array of shape (..., 28) containing the estimated model parameters.
        Element 0 is the natural logarithm of the estimated signal without
        diffusion-weighting, elements 1-6 are the estimated diffusion tensor
        elements in Voigt notation, and elements 7-27 are the estimated
        covariance tensor elements in Voigt notation.
    """
    params = np.zeros((np.product(mask.shape), 28)) * np.nan
    data_masked = data[mask]
    size = len(data_masked)
    if step >= size:  # Fit over all data simultaneously
        S = np.log(data_masked)[..., np.newaxis]
        C = data_masked[:, np.newaxis, :]
        B = X.T * C
        A = np.linalg.pinv(B @ X)
        params_masked = (A @ B @ S)[..., 0]
    else:  # Iterate over data
        params_masked = np.zeros((size, 28))
        for i in range(0, size, step):
            S = np.log(data_masked[i:i + step])[..., np.newaxis]
            C = data_masked[i:i + step][:, np.newaxis, :]
            B = X.T * C
            A = np.linalg.pinv(B @ X)
            params_masked[i:i + step] = (A @ B @ S)[..., 0]
    params[np.where(mask.ravel())] = params_masked
    params = params.reshape((mask.shape + (28,)))
    return params


class StandardModel(ReconstModel):

    def __init__(self, gtab, csf_comp=False, sh_order=6, fit_method='CSM'):
        """Standard model of diffusion in neuronal tissue [1]_.

        Parameters
        ----------
        gtab : dipy.core.gradients.GradientTable
            Gradient table with b-tensors.
        fit_method : str, optional
            Must be one of the followng:
                'CSM' for
                    :func:`qti._ols_fit`
                'WLS' for weighted least squares
                    :func:`qti._wls_fit`

        References
        ----------
        .. [1] Westin, Carl-Fredrik, et al. "Q-space trajectory imaging for
           multidimensional diffusion MRI of the human brain." Neuroimage 135
           (2016): 345-362. https://doi.org/10.1016/j.neuroimage.2016.02.039.
        """
        ReconstModel.__init__(self, gtab)

        if self.gtab.btens is None:
            raise ValueError(
                'QTI requires b-tensors to be defined in the gradient table.')
        self.X = design_matrix(self.gtab.btens)
        rank = np.linalg.matrix_rank(self.X.T @ self.X)
        if rank < 28:
            warn(
                'The combination of the b-tensor shapes, sizes, and ' +
                'orientations does not enable all elements of the covariance ' +
                'tensor to be estimated (rank(X.T @ X) = %s < 28).' % rank
            )

        try:
            self.fit_method = common_fit_methods[fit_method]
        except KeyError:
            raise ValueError(
                'Invalid value (%s) for \'fit_method\'.' % fit_method
                + ' Options: \'OLS\', \'WLS\'.'
            )

    def fit(self, data, mask=None):
        """Fit QTI to data.

        Parameters
        ----------
        data : numpy.ndarray
            Array of shape (..., number of acquisitions).
        mask : numpy.ndarray, optional
            Array with the same shape as the data array of a single acquisition.

        Returns
        -------
        qtifit : dipy.reconst.qti.QtiFit
            The fitted model.
        """
        if mask is None:
            mask = np.ones(data.shape[:-1], dtype=bool)
        else:
            if mask.shape != data.shape[:-1]:
                raise ValueError('Mask is not the same shape as data.')
            mask = np.array(mask, dtype=bool, copy=False)
        params = self.fit_method(data, mask, self.X)
        return QtiFit(params)

    def predict(self, params):
        """Generate signals from this model class instance and given parameters.

        Parameters
        ----------
        params : numpy.ndarray
            Array of shape (..., 28) containing the model parameters. Element 0
            is the natural logarithm of the signal without diffusion-weighting,
            elements 1-6 are the diffusion tensor elements in Voigt notation,
            and elements 7-27 are the covariance tensor elements in Voigt
            notation.

        Returns
        -------
        S : numpy.ndarray
            Signals.
        """
        S0 = np.exp(params[..., 0])
        D = params[..., 1:7, np.newaxis]
        C = params[..., 7::, np.newaxis]
        S = qti_signal(self.gtab, D, C, S0)
        return S


class QtiFit(object):

    def __init__(self, params):
        """Fitted QTI model.

        Parameters
        ----------
        params : numpy.ndarray
            Array of shape (..., 28) containing the model parameters. Element 0
            is the natural logarithm of the signal without diffusion-weighting,
            elements 1-6 are the diffusion tensor elements in Voigt notation,
            and elements 7-27 are the covariance tensor elements in Voigt
            notation.
        """
        self.params = params

    def predict(self, gtab):
        """Generate signals from this model fit and a given gradient table.

        Parameters
        ----------
        gtab : dipy.core.gradients.GradientTable
            Gradient table with b-tensors.

        Returns
        -------
        S : numpy.ndarray
            Signals.
        """
        if gtab.btens is None:
            raise ValueError(
                'QTI requires b-tensors to be defined in the gradient table.')
        S0 = self.S0_hat
        D = self.params[..., 1:7, np.newaxis]
        C = self.params[..., 7::, np.newaxis]
        S = qti_signal(gtab, D, C, S0)
        return S

    @auto_attr
    def S0_hat(self):
        """Estimated signal without diffusion-weighting.

        Returns
        -------
        S0 : numpy.ndarray
        """
        S0 = np.exp(self.params[..., 0])
        return S0

    @auto_attr
    def md(self):
        """Mean diffusivity.

        Returns
        -------
        md : numpy.ndarray

        Notes
        -----
        Mean diffusivity is calculated as

            .. math::

                \text{MD} = \langle \mathbf{D} \rangle : \mathbf{E}_\text{iso}
        """
        md = np.matmul(
            self.params[..., np.newaxis, 1:7],
            from_3x3_to_6x1(e_iso)
        )[..., 0, 0]
        return md

    @auto_attr
    def v_md(self):
        """Variance of microscopic mean diffusivities.

        Returns
        -------
        v_md : numpy.ndarray

        Notes
        -----
        Variance of microscopic mean diffusivities is calculated as

            .. math::

                V_\text{MD} = \mathbb{C} : \mathbb{E}_\text{bulk}
        """
        v_md = np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_bulk)
        )[..., 0, 0]
        return v_md

    @auto_attr
    def v_shear(self):
        """Shear variance.

        Returns
        -------
        v_shear : numpy.ndarray

        Notes
        -----
        Shear variance is calculated as

            .. math::

                V_\text{shear} = \mathbb{C} : \mathbb{E}_\text{shear}
        """
        v_shear = np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_shear)
        )[..., 0, 0]
        return v_shear

    @auto_attr
    def v_iso(self):
        """Total isotropic variance.

        Returns
        -------
        v_iso : numpy.ndarray

        Notes
        -----
        Total isotropic variance is calculated as

            .. math::

                V_\text{iso} = \mathbb{C} : \mathbb{E}_\text{iso}
        """
        v_iso = np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_iso)
        )[..., 0, 0]
        return v_iso

    @auto_attr
    def d_sq(self):
        """Diffusion tensor's outer product with itself.

        Returns
        -------
        d_sq : numpy.ndarray
        """
        d_sq = np.matmul(
            self.params[..., 1:7, np.newaxis],
            self.params[..., np.newaxis, 1:7]
        )
        return d_sq

    @auto_attr
    def mean_d_sq(self):
        """Average of microscopic diffusion tensors' outer products with
        themselves.

        Returns
        -------
        mean_d_sq : numpy.ndarray

        Notes
        -----
        Average of microscopic diffusion tensors' outer products with themselves
        is calculated as

            .. math::

                \langle \mathbf{D} \otimes \mathbf{D} \rangle = \mathbb{C} +
                \langle \mathbf{D} \rangle \otimes \langle \mathbf{D} \rangle
        """
        mean_d_sq = from_21x1_to_6x6(
            self.params[..., 7::, np.newaxis]) + self.d_sq
        return mean_d_sq

    @auto_attr
    def c_md(self):
        """Normalized variance of mean diffusivities.

        Returns
        -------
        c_md : numpy.ndarray

        Notes
        -----
        Normalized variance of microscopic mean diffusivities is calculated as

            .. math::

                C_\text{MD} = \frac{\mathbb{C} : \mathbb{E}_\text{bulk}}
                {\langle \mathbf{D} \otimes \mathbf{D} \rangle :
                \mathbb{E}_\text{bulk}}
        """
        c_md = self.v_md / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.mean_d_sq), -1, -2),
            from_6x6_to_21x1(E_bulk))[..., 0, 0]
        return c_md

    @auto_attr
    def c_mu(self):
        """Normalized microscopic anisotropy.

        Returns
        -------
        c_mu : numpy.ndarray

        Notes
        -----
        Normalized microscopic anisotropy is calculated as

            .. math::

                C_\mu = \frac{3}{2} \frac{\langle \mathbf{D} \otimes \mathbf{D}
                \rangle : \mathbb{E}_\text{shear}}{\langle \mathbf{D} \otimes
                \mathbf{D} \rangle : \mathbb{E}_\text{iso}}
        """
        c_mu = (1.5 * np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.mean_d_sq), -1, -2),
            from_6x6_to_21x1(E_shear)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.mean_d_sq), -1, -2),
            from_6x6_to_21x1(E_iso)))[..., 0, 0]
        return c_mu

    @auto_attr
    def ufa(self):
        """Microscopic fractional anisotropy.

        Returns
        -------
        ufa : numpy.ndarray

        Notes
        -----
        Microscopic fractional anisotropy is calculated as

            .. math::

                \mu\text{FA} = \sqrt{C_\mu}
        """
        ufa = np.sqrt(self.c_mu)
        return ufa

    @auto_attr
    def c_m(self):
        """Normalized macroscopic anisotropy.

        Returns
        -------
        c_m : numpy.ndarray

        Notes
        -----
        Normalized macroscopic anisotropy is calculated as

            .. math::

                C_\text{M} = \frac{3}{2} \frac{\langle \mathbf{D} \rangle
                \otimes \langle \mathbf{D} \rangle : \mathbb{E}_\text{shear}}
                {\langle \mathbf{D} \rangle \otimes \langle \mathbf{D} \rangle :
                \mathbb{E}_\text{iso}}
        """
        c_m = (1.5 * np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_shear)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_iso)))[..., 0, 0]
        return c_m

    @auto_attr
    def fa(self):
        """Fractional anisotropy.

        Returns
        -------
        fa : numpy.ndarray

        Notes
        -----
        Fractional anisotropy is calculated as

            .. math::

                \text{FA} = \sqrt{C_\text{M}}
        """
        fa = np.sqrt(self.c_m)
        return fa

    @auto_attr
    def c_c(self):
        """Microscopic orientation coherence.

        Returns
        -------
        c_c : numpy.ndarray

        Notes
        -----
        Microscopic orientation coherence is calculated as

            .. math::

                C_c = \frac{C_\text{M}}{C_\mu}
        """
        c_c = self.c_m / self.c_mu
        return c_c

    @auto_attr
    def mk(self):
        """Mean kurtosis.

        Returns
        -------
        mk : numpy.ndarray

        Notes
        -----
        Mean kurtosis is calculated as

            .. math::

                \text{MK} = K_\text{bulk} + K_\text{shear}
        """
        mk = self.k_bulk + self.k_shear
        return mk

    @auto_attr
    def k_bulk(self):
        """Bulk kurtosis.

        Returns
        -------
        k_bulk : numpy.ndarray

        Notes
        -----
        Bulk kurtosis is calculated as

            .. math::

                K_\text{bulk} = 3 \frac{\mathbb{C} : \mathbb{E}_\text{bulk}}
                {\langle \mathbf{D} \rangle \otimes \langle \mathbf{D} \rangle :
                \mathbb{E}_\text{bulk}}
        """
        k_bulk = (3 * np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_bulk)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_bulk)))[..., 0, 0]
        return k_bulk

    @auto_attr
    def k_shear(self):
        """Shear kurtosis.

        Returns
        -------
        k_shear : numpy.ndarray

        Notes
        -----
        Shear kurtosis is calculated as

            .. math::

                K_\text{shear} = \frac{6}{5} \frac{\mathbb{C} :
                \mathbb{E}_\text{shear}}{\langle \mathbf{D} \rangle \otimes
                \langle \mathbf{D} \rangle : \mathbb{E}_\text{bulk}}
        """
        k_shear = (6 / 5 * np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_shear)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_bulk)))[..., 0, 0]
        return k_shear

    @auto_attr
    def k_mu(self):
        """Microscopic kurtosis.

        Returns
        -------
        k_mu : numpy.ndarray

        Notes
        -----
        Microscopic kurtosis is calculated as

            .. math::

                K_\mu = \frac{6}{5} \frac{\langle \mathbf{D} \otimes \mathbf{D}
                \rangle : \mathbb{E}_\text{shear}}{\langle \mathbf{D} \rangle
                \otimes \langle \mathbf{D} \rangle : \mathbb{E}_\text{bulk}}
        """
        k_mu = (6 / 5 * np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.mean_d_sq), -1, -2),
            from_6x6_to_21x1(E_shear)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_bulk)))[..., 0, 0]
        return k_mu


common_fit_methods = {'OLS': _ols_fit, 'WLS': _wls_fit}
