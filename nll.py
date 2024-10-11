import numpy as np
from scipy import special
import gf


def nll(s_model, s_meas, sigma):
    """
    Compute the negative log-likelihood (nll) using a Ricean model of the MR
    signal i.e. the likelihood is the product over N Ricean distributions
    (where N is the number of echoes). Therefore log(likelihood) is the sum
    over N log(Ricean distributions)

    :param s_model: model signal
    :type s_model: np.ndarray
    :param s_meas: measured signal
    :type s_meas: np.ndarray
    :param sigma: standard deviation of Gaussian noise on in-phase and
    quadrature channels
    :type sigma: float
    :return: negative log-likelihood
    :rtype: float
    """

    # Ricean PDF: P = (s_meas_1 / sigma ** 2) * np.exp(-(s_meas_1 ** 2 + s_ideal ** 2) / (2 * sigma ** 2)) * special.i0(z) where z = s_meas_1 * s_ideal / sigma ** 2
    # Since special.i0(z) overflows when z is large we use the exponentially scaled version instead: special.i0e(z)
    # Given that special.i0e(z) = special.i0(z) * np.exp(-np.abs(z.real)) we can rearrange to find special.i0(z) = special.i0e(z) * np.exp(np.abs(z.real))
    # Then taking logs: np.log(special.i0(z)) = np.log)(special.i0e(z)) + np.abs(z.real)
    # So log(P) is: np.log(s_meas) - 2 * np.log(sigma) - ((s_meas ** 2 + s_model ** 2) / (2 * sigma ** 2)) + np.log(special.i0e(z)) + np.abs(z.real)
    # note this is equivalant to -stats.rice.logpdf(s_meas[ii], s_model[ii] / sigma, scale=sigma)
    z = s_meas * s_model / sigma**2
    neglogParray = -(
        np.log(s_meas)
        - 2 * np.log(sigma)
        - ((s_meas**2 + s_model**2) / (2 * sigma**2))
        + np.log(special.i0e(z))
        + np.abs(z.real)
    )

    # sum logP array to give negative log likelihood and then replace any infs with large numbers
    negLogLik = np.nan_to_num(np.sum(neglogParray))

    return negLogLik


def nll_gf(p, K, t1, tau, n, fa_ex, fa_ref, s_meas):
    """
    Compute the negative log-likelihood (nll) using a Ricean model of the MR
    signal when estimating A, t2, b1 and sigma using generating function to
    calculate the signal

    :param p: parameters to fit [A, t2, b1, sigma]
    :type p: np.ndarray
    :param K: the number of points on the unit circle at which the generating function is calculated
    :type K: int
    :param t1: T1 of water (s)
    :type t1: float
    :param tau: echo spacing in s
    :type tau: float
    :param n: number of echoes
    :type n: int
    :param fa_ex: array of excitation flip angles to account for an imperfect slice profile
    :type fa_ex: np.ndarray[float]
    :param fa_ref: array of refocussing flip angles to account for an imperfect slice profile
    :type fa_ref: np.ndarray[float]
    :param s_meas: measured signal
    :type s_meas: np.ndarray
    :return: negative log-likelihood
    :rtype: float
    """

    A = p[0]
    t2 = p[1]
    b1 = p[2]
    sigma = p[3]

    s_model = gf.GF(A, K, t1, t2, tau, n, fa_ex, fa_ref, b1)

    return nll(s_model, s_meas, sigma)
