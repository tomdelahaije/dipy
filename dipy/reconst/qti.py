"""Classes and functions for fitting the covariance tensor model of q-space
trajectory imaging (QTI) by Westin et al. as presented in “Q-space trajectory
imaging for multidimensional diffusion MRI of the human brain” NeuroImage vol.
135 (2016): 345-62. https://doi.org/10.1016/j.neuroimage.2016.02.039"""

from warnings import warn

import numpy as np

from packaging.version import Version

from dipy.core.optimize import PositiveDefiniteLeastSquares
from dipy.data import load_sdp_constraints
from dipy.reconst.base import ReconstModel
from dipy.reconst.dti import auto_attr
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.utils.optpkg import optional_package

cp, have_cvxpy, _ = optional_package("cvxpy")


# XXX Eventually to be replaced with `reconst.dti.lower_triangular`
def from_3x3_to_6x1(T):
    """Convert symmetric 3 x 3 matrices into 6 x 1 vectors.

    Parameters
    ----------
    T : numpy.ndarray
        An array of size (..., 3, 3).

    Returns
    -------
    V : numpy.ndarray
        Converted vectors of size (..., 6, 1).

    Notes
    -----
    The conversion of a matrix into a vector is defined as

        .. math::

            \mathbf{V} = \begin{bmatrix}
            T_{11} & T_{22} & T_{33} &
            \sqrt{2} T_{23} & \sqrt{2} T_{13} & \sqrt{2} T_{12}
            \end{bmatrix}^T
    """
    if T.shape[-2::] != (3, 3):
        raise ValueError('The shape of the input array must be (..., 3, 3).')
    if not np.all(np.isclose(T, np.swapaxes(T, -1, -2))):
        warn('All matrices converted to Voigt notation are not symmetric.')
    C = np.sqrt(2)
    V = np.stack((T[..., 0, 0],
                  T[..., 1, 1],
                  T[..., 2, 2],
                  C * T[..., 1, 2],
                  C * T[..., 0, 2],
                  C * T[..., 0, 1]), axis=-1)[..., np.newaxis]
    return V


def from_6x1_to_3x3(V):
    """Convert 6 x 1 vectors into symmetric 3 x 3 matrices.

    Parameters
    ----------
    V : numpy.ndarray
        An array of size (..., 6, 1).

    Returns
    -------
    T : numpy.ndarray
        Converted matrices of size (..., 3, 3).

    Notes
    -----
    The conversion of a matrix into a vector is defined as

        .. math::

            \mathbf{V} = \begin{bmatrix}
            T_{11} & T_{22} & T_{33} &
            \sqrt{2} T_{23} & \sqrt{2} T_{13} & \sqrt{2} T_{12}
            \end{bmatrix}^T
    """
    if V.shape[-2::] != (6, 1):
        raise ValueError('The shape of the input array must be (..., 6, 1).')
    C = 1 / np.sqrt(2)
    T = np.array(([V[..., 0, 0], C * V[..., 5, 0], C * V[..., 4, 0]],
                  [C * V[..., 5, 0], V[..., 1, 0], C * V[..., 3, 0]],
                  [C * V[..., 4, 0], C * V[..., 3, 0], V[..., 2, 0]]))
    T = np.moveaxis(T, (0, 1), (-2, -1))
    return T


def from_6x6_to_21x1(T):
    """Convert symmetric 6 x 6 matrices into 21 x 1 vectors.

    Parameters
    ----------
    T : numpy.ndarray
        An array of size (..., 6, 6).

    Returns
    -------
    V : numpy.ndarray
        Converted vectors of size (..., 21, 1).

    Notes
    -----
    The conversion of a matrix into a vector is defined as

        .. math::

            \begin{matrix}
            \mathbf{V} = & \big[
            T_{11} & T_{22} & T_{33} \\
            & \sqrt{2} T_{23} & \sqrt{2} T_{13} & \sqrt{2} T_{12} \\
            & \sqrt{2} T_{14} & \sqrt{2} T_{15} & \sqrt{2} T_{16} \\
            & \sqrt{2} T_{24} & \sqrt{2} T_{25} & \sqrt{2} T_{26} \\
            & \sqrt{2} T_{34} & \sqrt{2} T_{35} & \sqrt{2} T_{36} \\
            & T_{44} & T_{55} & T_{66} \\
            & \sqrt{2} T_{45} & \sqrt{2} T_{56} & \sqrt{2} T_{46} \big]^T
            \end{matrix}
    """
    if T.shape[-2::] != (6, 6):
        raise ValueError('The shape of the input array must be (..., 6, 6).')
    if not np.all(np.isclose(T, np.swapaxes(T, -1, -2), equal_nan=True)):
        warn('All matrices converted to Voigt notation are not symmetric.')
    C = np.sqrt(2)
    V = np.stack(([T[..., 0, 0], T[..., 1, 1], T[..., 2, 2],
                   C * T[..., 1, 2], C * T[..., 0, 2], C * T[..., 0, 1],
                   C * T[..., 0, 3], C * T[..., 0, 4], C * T[..., 0, 5],
                   C * T[..., 1, 3], C * T[..., 1, 4], C * T[..., 1, 5],
                   C * T[..., 2, 3], C * T[..., 2, 4], C * T[..., 2, 5],
                   T[..., 3, 3], T[..., 4, 4], T[..., 5, 5],
                   C * T[..., 3, 4], C * T[..., 4, 5], C * T[..., 3, 5]]),
                 axis=-1)[..., np.newaxis]
    return V


def from_21x1_to_6x6(V):
    """Convert 21 x 1 vectors into symmetric 6 x 6 matrices.

    Parameters
    ----------
    V : numpy.ndarray
        An array of size (..., 21, 1).

    Returns
    -------
    T : numpy.ndarray
        Converted matrices of size (..., 6, 6).

    Notes
    -----
    The conversion of a matrix into a vector is defined as

        .. math::

            \begin{matrix}
            \mathbf{V} = & \big[
            T_{11} & T_{22} & T_{33} \\
            & \sqrt{2} T_{23} & \sqrt{2} T_{13} & \sqrt{2} T_{12} \\
            & \sqrt{2} T_{14} & \sqrt{2} T_{15} & \sqrt{2} T_{16} \\
            & \sqrt{2} T_{24} & \sqrt{2} T_{25} & \sqrt{2} T_{26} \\
            & \sqrt{2} T_{34} & \sqrt{2} T_{35} & \sqrt{2} T_{36} \\
            & T_{44} & T_{55} & T_{66} \\
            & \sqrt{2} T_{45} & \sqrt{2} T_{56} & \sqrt{2} T_{46} \big]^T
            \end{matrix}
    """
    if V.shape[-2::] != (21, 1):
        raise ValueError('The shape of the input array must be (..., 21, 1).')
    C = 1 / np.sqrt(2)
    T = np.array((
        [    V[...,  0, 0], C * V[...,  5, 0], C * V[...,  4, 0],
         C * V[...,  6, 0], C * V[...,  7, 0], C * V[...,  8, 0]],
        [C * V[...,  5, 0],     V[...,  1, 0], C * V[...,  3, 0],
         C * V[...,  9, 0], C * V[..., 10, 0], C * V[..., 11, 0]],
        [C * V[...,  4, 0], C * V[...,  3, 0],     V[...,  2, 0],
         C * V[..., 12, 0], C * V[..., 13, 0], C * V[..., 14, 0]],
        [C * V[...,  6, 0], C * V[...,  9, 0], C * V[..., 12, 0],
             V[..., 15, 0], C * V[..., 18, 0], C * V[..., 20, 0]],
        [C * V[...,  7, 0], C * V[..., 10, 0], C * V[..., 13, 0],
         C * V[..., 18, 0],     V[..., 16, 0], C * V[..., 19, 0]],
        [C * V[...,  8, 0], C * V[..., 11, 0], C * V[..., 14, 0],
         C * V[..., 20, 0], C * V[..., 19, 0],     V[..., 17, 0]]
    ))
    T = np.moveaxis(T, (0, 1), (-2, -1))
    return T


def cvxpy_1x6_to_3x3(V):
    """Convert a 1 x 6 vector into a symmetric 3 x 3 matrix.

    Parameters
    ----------
    V : numpy.ndarray
        An array of size (1, 6).

    Returns
    -------
    T : cvxpy.bmat
        Converted matrix of size (3, 3).

    Notes
    -----
    The conversion of a matrix into a vector is defined as

        .. math::

            \mathbf{V} = \begin{bmatrix}
            T_{11} & T_{22} & T_{33} &
            \sqrt{2} T_{23} & \sqrt{2} T_{13} & \sqrt{2} T_{12}
            \end{bmatrix}^T
    """
    if V.shape[0] == 6:
        V = V.T

    f = 1 / np.sqrt(2)

    T = cp.bmat([[V[0, 0], f * V[0, 5], f * V[0, 4]],
                 [f * V[0, 5],     V[0, 1], f * V[0, 3]],
                 [f * V[0, 4], f * V[0, 3],    V[0, 2]]])
    return T


def cvxpy_1x21_to_6x6(V):
    """Convert 1 x 21 vector into a symmetric 6 x 6 matrix.

    Parameters
    ----------
    V : numpy.ndarray
        An array of size (1, 21).

    Returns
    -------
    T : cvxpy.bmat
        Converted matrices of size (6, 6).

    Notes
    -----
    The conversion of a matrix into a vector is defined as

        .. math::

            \begin{matrix}
            \mathbf{V} = & \big[
            T_{11} & T_{22} & T_{33} \\
            & \sqrt{2} T_{23} & \sqrt{2} T_{13} & \sqrt{2} T_{12} \\
            & \sqrt{2} T_{14} & \sqrt{2} T_{15} & \sqrt{2} T_{16} \\
            & \sqrt{2} T_{24} & \sqrt{2} T_{25} & \sqrt{2} T_{26} \\
            & \sqrt{2} T_{34} & \sqrt{2} T_{35} & \sqrt{2} T_{36} \\
            & T_{44} & T_{55} & T_{66} \\
            & \sqrt{2} T_{45} & \sqrt{2} T_{56} & \sqrt{2} T_{46} \big]^T
            \end{matrix}
    """
    if V.shape[0] == 21:
        V = V.T

    f = 1 / np.sqrt(2)

    T = cp.bmat([[V[0, 0], f * V[0, 5], f * V[0, 4], f * V[0, 6],
                f * V[0, 7], f * V[0, 8]],
                [f * V[0, 5], V[0, 1],
                 f * V[0, 3], f * V[0, 9], f * V[0, 10], f * V[0, 11]],
                [f * V[0, 4], f * V[0, 3], V[0, 2], f * V[0, 12],
                 f * V[0, 13], f * V[0, 14]],
                [f * V[0, 6], f * V[0, 9],
                 f * V[0, 12], V[0, 15], f * V[0, 18], f * V[0, 20]],
                [f * V[0, 7], f * V[0, 10], f * V[0, 13], f * V[0, 18],
                 V[0, 16], f * V[0, 19]],
                [f * V[0, 8], f * V[0, 11],
                 f * V[0, 14], f * V[0, 20], f * V[0, 19], V[0, 17]]])
    return T


# These tensors are used in the calculation of the QTI parameters
e_iso = np.eye(3) / 3
E_iso = np.eye(6) / 3
E_bulk = from_3x3_to_6x1(e_iso) @ from_3x3_to_6x1(e_iso).T
E_shear = E_iso - E_bulk
E_tsym = E_bulk + .4 * E_shear


def dtd_covariance(DTD):
    """Calculate covariance of a diffusion tensor distribution (DTD).

    Parameters
    ----------
    DTD : numpy.ndarray
        Diffusion tensor distribution of shape (number of tensors, 3, 3) or
        (number of tensors, 6, 1).

    Returns
    -------
    C : numpy.ndarray
        Covariance tensor of shape (6, 6).

    Notes
    -----
    The covariance tensor is calculated according to the following equation and
    converted into a rank-2 tensor [1]_:

        .. math::

            \mathbb{C} = \langle \mathbf{D} \otimes \mathbf{D} \rangle -
            \langle \mathbf{D} \rangle \otimes \langle \mathbf{D} \rangle

    References
    ----------
    .. [1] Westin, Carl-Fredrik, et al. "Q-space trajectory imaging for
       multidimensional diffusion MRI of the human brain." Neuroimage 135
       (2016): 345-362. https://doi.org/10.1016/j.neuroimage.2016.02.039.
    """
    dims = DTD.shape
    if len(dims) != 3 or (dims[1:3] != (3, 3) and dims[1:3] != (6, 1)):
        raise ValueError(
            'The shape of DTD must be (number of tensors, 3, 3) or (number of '
            + 'tensors, 6, 1).')
    if dims[1:3] == (3, 3):
        DTD = from_3x3_to_6x1(DTD)
    D = np.mean(DTD, axis=0)
    C = (np.mean(DTD @ np.swapaxes(DTD, -2, -1), axis=0)
         - D @ np.swapaxes(D, -2, -1))
    return C


def qti_signal(gtab, D, C, S0=1):
    """Generate signals using the covariance tensor signal representation.

    Parameters
    ----------
    gtab : dipy.core.gradients.GradientTable
        Gradient table with b-tensors.
    D : numpy.ndarray
        Diffusion tensors of shape (..., 3, 3), (..., 6, 1), or (..., 6).
    C : numpy.ndarray
        Covariance tensors of shape (..., 6, 6), (..., 21, 1), or (..., 21).
    S0 : numpy.ndarray, optional
        Signal magnitudes without diffusion-weighting. Must be a single number
        or an array of same shape as D and C without the last two dimensions.

    Returns
    -------
    S : numpy.ndarray
        Simulated signals.

    Notes
    -----
    The signal is generated according to

        .. math::

            S = S_0 \exp \left(- \mathbf{b} : \langle \mathbf{D} \rangle
            + \frac{1}{2}(\mathbf{b} \otimes \mathbf{b}) : \mathbb{C} \right)
    """

    # Validate input and convert to Voigt notation if necessary
    if gtab.btens is None:
        raise ValueError(
            'QTI requires b-tensors to be defined in the gradient table.')
    if D.shape[-2::] != (6, 1):
        if D.shape[-2::] == (3, 3):
            D = from_3x3_to_6x1(D)
        elif D.shape[-1] == 6:
            D = D[..., np.newaxis]
        else:
            raise ValueError(
                'The shape of D must be (..., 3, 3), (..., 6, 1) or (..., 6).')
    if C.shape[-2::] != (21, 1):
        if C.shape[-2::] == (6, 6):
            C = from_6x6_to_21x1(C)
        elif C.shape[-1] == 21:
            C = C[..., np.newaxis]
        else:
            raise ValueError(
                'The shape of C must be (..., 6, 6), (..., 21, 1), or '
                + '(..., 21).'
            )
    if D.shape[0:-2] != C.shape[0:-2]:
        raise ValueError('The shapes of C and D are not compatible')
    if not (isinstance(S0, int) or isinstance(S0, float)):
        if S0.shape != (1,) and S0.shape != D.shape[0:-2]:
            raise ValueError(
                'S0 must be a single number or an array of the same shape '
                + ' compatible with D and C.'
            )

    # Generate signals
    S = np.zeros(D.shape[0:-2] + (gtab.btens.shape[0],))
    for i, bten in enumerate(gtab.btens):
        b = from_3x3_to_6x1(bten)
        b_sq = from_6x6_to_21x1(b @ np.swapaxes(b, -2, -1))
        S[..., i] = S0 * np.exp(
            - np.swapaxes(b, -2, -1) @ D
            + .5 * np.swapaxes(b_sq, -2, -1) @ C)[..., 0, 0]
    return S


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
        X[i] = np.concatenate(([1], (-b.T)[0, :], (0.5 * b_sq.T)[0, :]))
    return X


def _ols_fit(data, X, Xinv):
    """Estimate the model parameters using ordinary least squares.

    Parameters
    ----------
    data : numpy.ndarray
        Signal array of shape (number of acquisitions).
    X : numpy.ndarray
        Design matrix of shape (number of acquisitions, 28).
    Xinv : numpy.ndarray
        Inverse of the design matrix.

    Returns
    -------
    params : numpy.ndarray
        Array of shape (..., 28) containing the estimated model parameters.
        Element 0 is the natural logarithm of the estimated signal without
        diffusion-weighting, elements 1-6 are the estimated diffusion tensor
        elements in Voigt notation, and elements 7-27 are the estimated
        covariance tensor elements in Voigt notation.
    """
    params = Xinv @ np.log(data)
    return params


def _wls_fit(data, X, Xinv):
    """Estimate the model parameters using weighted least squares with the
    signal magnitudes as weights.

    Parameters
    ----------
    data : numpy.ndarray
        Signal array of shape (number of acquisitions).
    X : numpy.ndarray
        Design matrix of shape (number of acquisitions, 28).
    Xinv : numpy.ndarray
        Inverse of the design matrix.

    Returns
    -------
    params : numpy.ndarray
        Array of shape (..., 28) containing the estimated model parameters.
        Element 0 is the natural logarithm of the estimated signal without
        diffusion-weighting, elements 1-6 are the estimated diffusion tensor
        elements in Voigt notation, and elements 7-27 are the estimated
        covariance tensor elements in Voigt notation.
    """
    B = X.T * data
    A = np.linalg.pinv(B @ X)
    params = A @ B @ np.log(data)
    return params


def _sdpdc_fit(data, X, cvxpy_solver=None):
    """Estimate the model parameters using Semidefinite Programming (SDP),
    while enforcing positivity constraints on the D and C tensors (SDPdc) [2]_

    Parameters
    ----------
    data : numpy.ndarray
        Signal array of shape (number of acquisitions).
    X : numpy.ndarray
        Design matrix of shape (number of acquisitions, 28).
    cvxpy_solver: string, optional
        The name of the SDP solver to be used. Default: None

    Returns
    -------
    params : numpy.ndarray
        Array of shape (..., 28) containing the estimated model parameters.
        Element 0 is the natural logarithm of the estimated signal without
        diffusion-weighting, elements 1-6 are the estimated diffusion tensor
        elements in Voigt notation, and elements 7-27 are the estimated
        covariance tensor elements in Voigt notation.

    References
    ----------
    .. [2] Herberthson M., Boito D., Dela Haije T., Feragen A., Westin C.-F.,
           Ozarslan E., "Q-space trajectory imaging with positivity constraints
           (QTI+)" in Neuroimage, Volume 238, 2021.
    """
    scale = np.max(data)
    scale = scale if scale > 1. else 1.
    data_scaled = data / scale
    log_data = np.asarray([np.log(data_scaled)]).T

    W = np.diag(data_scaled)
    y = W @ log_data
    A = W @ X

    x = cp.Variable((28, 1))
    dc = cvxpy_1x6_to_3x3(x[1:7])
    cc = cvxpy_1x21_to_6x6(x[7:])
    constraints = [dc >> 0, cc >> 0]
    if Version(cp.__version__) < Version('1.1'):
        objective = cp.Minimize(cp.norm(A * x - y))
    else:
        objective = cp.Minimize(cp.norm(A @ x - y))
    prob = cp.Problem(objective, constraints)
    unconstrained = cp.Problem(objective)

    try:
        prob.solve(solver=cvxpy_solver, verbose=False)
        params = np.squeeze(x.value)
    except Exception:
        msg = 'Constrained optimization failed, attempting unconstrained'
        msg += ' optimization.'
        warn(msg)
        try:
            unconstrained.solve(solver=cvxpy_solver)
            params = np.squeeze(x.value)
        except Exception:
            msg = 'Unconstrained optimization failed,'
            msg += ' returning zero array.'
            warn(msg)
            return np.zeros(x.shape)

    params[0] += np.log(scale)

    return params


def _cls_fit(sdp, data, X, cvxpy_solver=None):
    """Estimate the model parameters using ordinary least squares with convexity
    constraints.

    Parameters
    ----------
    sdp : PositiveDefiniteLeastSquares instance
        A CVXPY representation of a regularized least squares optimization
        problem.
    data : numpy.ndarray
        Array of shape (..., number of acquisitions).
    mask : numpy.ndarray
        Boolean array with the same shape as the data array of a single
        acquisition.
    X : numpy.ndarray
        Design matrix of shape (number of acquisitions, 28).

    Returns
    -------
    params : numpy.ndarray
        Array of shape (..., 28) containing the estimated model parameters.
        Element 0 is the natural logarithm of the estimated signal without
        diffusion-weighting, elements 1-6 are the estimated diffusion tensor
        elements in Voigt notation, and elements 7-27 are the estimated
        covariance tensor elements in Voigt notation.
    """
    params = sdp.solve(X, np.log(data), solver=cvxpy_solver)
    return params


def _cwls_fit(sdp, data, X, cvxpy_solver=None):
    """Estimate the model parameters using weighted least squares with convexity
    constraints.

    Parameters
    ----------
    sdp : PositiveDefiniteLeastSquares instance
        A CVXPY representation of a regularized least squares optimization
        problem.
    data : numpy.ndarray
        Array of shape (..., number of acquisitions).
    mask : numpy.ndarray
        Boolean array with the same shape as the data array of a single
        acquisition.
    X : numpy.ndarray
        Design matrix of shape (number of acquisitions, 28).

    Returns
    -------
    params : numpy.ndarray
        Array of shape (..., 28) containing the estimated model parameters.
        Element 0 is the natural logarithm of the estimated signal without
        diffusion-weighting, elements 1-6 are the estimated diffusion tensor
        elements in Voigt notation, and elements 7-27 are the estimated
        covariance tensor elements in Voigt notation.
    """
    W = np.diag(data)
    S = np.dot(W, np.log(data))
    params = sdp.solve(np.dot(W, X), S, solver=cvxpy_solver)
    return params


class QtiModel(ReconstModel):

    def __init__(self, gtab, fit_method='WLS', cvxpy_solver=None):
        """Covariance tensor model of q-space trajectory imaging [1]_.

        Parameters
        ----------
        gtab : dipy.core.gradients.GradientTable
            Gradient table with b-tensors.
        fit_method : str, optional
            Must be one of the following:
                'OLS' for ordinary least squares
                    :func:`qti._ols_fit`
                'WLS' for weighted least squares
                    :func:`qti._wls_fit`
                'SDPDc' for semidefinite programming with positivity constraints
                    applied [2]_
                    :func:`qti._sdpdc_fit`
                'CLS' for ordinary least squares with convexity constraints
                    :func:`qti._cls_fit`
                'CWLS' for weighted least squares with convexity constraints
                    :func:`qti._cwls_fit`
        cvxpy_solver: str, optional
            solver for the SDP formulation. default: None

        References
        ----------
        .. [1] Westin, Carl-Fredrik, et al. "Q-space trajectory imaging for
           multidimensional diffusion MRI of the human brain." Neuroimage 135
           (2016): 345-362. https://doi.org/10.1016/j.neuroimage.2016.02.039.
        .. [2] Herberthson M., Boito D., Dela Haije T., Feragen A., Westin CF.,
           Ozarslan E., "Q-space trajectory imaging with positivity constraints
           (QTI+)" in Neuroimage, Volume 238, 2021.
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

        self.Xinv = np.linalg.pinv(self.X)

        try:
            self.fit_method = common_fit_methods[fit_method]
        except KeyError:
            raise ValueError(
                'Invalid value (%s) for \'fit_method\'.' % fit_method
                + " Options: 'OLS', 'WLS', 'SDPdc', 'CLS', 'CWLS'."
            )
        self.fit_method_name = fit_method

        self.convexity_constraint = fit_method in {'CLS', 'CWLS'}
        if self.convexity_constraint:
            self.sdp_constraints = load_sdp_constraints('qti')

        self.sdpdc_constraint = fit_method == 'SDPdc'
        if self.sdpdc_constraint:
            self.sdp_constraints = load_sdp_constraints('qti_dc')

        self.sdp_constraint = self.sdpdc_constraint or self.convexity_constraint
        if self.sdp_constraint:
            if not have_cvxpy:
                raise ValueError('CVXPY package needed to enforce constraints.')
            if cvxpy_solver is not None:
                if cvxpy_solver not in cp.installed_solvers():
                    msg = "Input `cvxpy_solver` was set to %s." % cvxpy_solver
                    msg += " One of %s" % ', '.join(cp.installed_solvers())
                    msg += " was expected."
                    raise ValueError(msg)
            self.cvxpy_solver = cvxpy_solver
            self.sdp = PositiveDefiniteLeastSquares(28, A=self.sdp_constraints)

    @multi_voxel_fit
    def fit(self, data):
        # if self.sdpdc_constraint:
        #     params = self.fit_method(data, self.X,
        #                              cvxpy_solver=self.cvxpy_solver)
        if self.sdp_constraint:
            params = self.fit_method(self.sdp, data, self.X,
                                     cvxpy_solver=self.cvxpy_solver)
        else:
            params = self.fit_method(data, self.X, self.Xinv)
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


common_fit_methods = {'OLS': _ols_fit, 'WLS': _wls_fit, 'SDPdc': _cwls_fit,
                      'CLS': _cls_fit, 'CWLS': _cwls_fit}
