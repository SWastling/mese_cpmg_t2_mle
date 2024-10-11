import numpy as np
import matplotlib.pyplot as plt


def GF(A, K, t1, t2, tau, n, fa_ex, fa_ref, b1, returnte0=False, debug=False):
    """
    Generating function as described in equation 1 and 2 of Petrovic et al 2015
    MRM 73:818-827 (2015) extended to allow:
    - an array of excitation flip angles to account for imperfect signal excitation
    - an RF scaling factor to account for B1 transmit inhomogeneity

    :param A: signal scale factor (i.e. signal at TE=0 is A)
    :type A: float
    :param K: the number of points on the unit circle at which the generating function is calculated
    :type K: int
    :param t1: longitudinal relaxtion time in s
    :type t1: float
    :param t2: transverse relaxtion time in s
    :type t2: float
    :param tau: echo spacing in s
    :type tau: float
    :param n: number of echoes
    :type n: int
    :param fa_ex: array of excitation flip angles to account for an imperfect slice profile
    :type fa_ex: np.ndarray[float]
    :param fa_ref: array of refocussing flip angles to account for an imperfect slice profile
    :type fa_ref: np.ndarray[float]
    :param b1: RF scale factor (between 0 and 1) to account for RF transmit inhomogeneity
    :type b1: float
    :param returnte0: include signal at TE=0 in output
    :type returnte0: bool
    :param debug: show output as plot if true
    :type debug: bool
    :return: signal as a function of echo time
    :rtype: np.ndarray[float]
    """

    # array of K points on unit circle in complex plane
    z = np.exp(1j * np.linspace(-np.pi, np.pi, K, endpoint=False))

    # relaxation terms
    e1 = np.exp(-tau / t1)
    e2 = np.exp(-tau / t2)

    # scale fa_ex and fa_ref by b1
    fa_ex = b1 * fa_ex
    fa_ref = b1 * fa_ref

    # Check if fa_ref is a single refocusing flip-angle or a profile
    # of flip-angles across the slice
    if isinstance(fa_ref, float):
        Q = 1
    else:
        Q = len(fa_ref)

    if Q > 1:
        # replicate z, fa_ex and fa_ref to enable calulation of signal at
        # multiple flip angles simultaneously i.e. across the slice profile
        z = np.transpose(np.tile(z, (Q, 1)))
        fa_ex = np.tile(fa_ex, (K, 1))
        fa_ref = np.tile(fa_ref, (K, 1))

    # calculate generating function (see equation 1 in Petrovic 2015)
    f = (np.sin(fa_ex) / 2) * (
        1
        + np.sqrt(
            ((1 + z * e2) * (1 - z * (e1 + e2) * np.cos(fa_ref) + z**2 * e1 * e2))
            / ((-1 + z * e2) * (-1 + z * (e1 - e2) * np.cos(fa_ref) + z**2 * e1 * e2))
        )
    )

    # combine across the slice profile
    norm_ex = np.mean(np.sin(fa_ex))
    if Q > 1:
        fsp = np.sum(f, axis=1) / (Q * norm_ex)
    else:
        fsp = f / norm_ex

    # determine the signal using the discrete fourier transform (equation 3 in
    # Petrovic 2015)
    s_slice = np.fft.fft(fsp)

    # select sampled echoes
    s = A * np.abs(s_slice) / K

    if returnte0:
        s = s[0 : n + 1]
    else:
        s = s[1 : n + 1]

    if debug:  # pragma: no cover
        plt.plot(abs(f))
        plt.title("GF output")
        plt.xlabel("z")
        plt.ylabel("$|F|$")
        plt.show()

        plt.plot(abs(fsp))
        plt.title("GF output")
        plt.xlabel("z")
        plt.ylabel("$|F_sp|$")
        plt.show()

        if returnte0:
            te = 1000 * np.arange(0, n + 1) * tau
        else:
            te = 1000 * np.arange(1, n + 1) * tau

        plt.plot(te, s)
        plt.title("GF output")
        plt.xlabel("Echo time (ms)")
        plt.ylabel("Signal (a.u.)")
        plt.show()

    return s
