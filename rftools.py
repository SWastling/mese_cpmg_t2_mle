import numpy as np
import matplotlib.pyplot as plt


def abr(rf, x, g=None):
    """
    Compute the Cayley-Klein parameters for the rotation produced by an
    RF pulse i.e. perform a forward SLR transform.

    Based on c-code by John Pauly (http://rsl.stanford.edu/research/software.html)

    :param rf: n point rf waveform
    :type rf: np.ndarray[complex]
    :param x: vectors of spatial positions to compute a and b
    :type x: np.ndarray[float]
    :param g: optional gradient waveform
    :type g: np.ndarray[float]
    :return: alpha and beta the Cayley-Klein parameters
    :rtype: tuple[np.ndarray[complex], np.ndarray[complex]]
    """

    if not g:
        # constant gradient with area = 2 * pi
        g = np.ones(len(rf)) * 2 * np.pi / len(rf)

    alpha = np.empty(len(x), dtype=complex)
    beta = np.empty_like(alpha)

    for ix, xloc in enumerate(x):
        a = complex(1.0, 0.0)
        b = complex(0.0, 0.0)
        for k, rfval in enumerate(rf):
            cg = xloc * g[k]
            cpr = np.real(rfval)
            cpi = np.imag(rfval)

            phi = np.sqrt(cg**2 + cpr * cpr + cpi * cpi)

            if phi > 0.0:
                nx = cpr / phi
                ny = cpi / phi
                nz = cg / phi
            else:
                nx = 0.0
                ny = 0.0
                nz = 1.0  # doesn't matter, phi=0

            al = complex(np.cos(phi / 2), nz * np.sin(phi / 2))
            be = complex(ny * np.sin(phi / 2), nx * np.sin(phi / 2))

            ap = complex(
                -(be.real * b.real - (-be.imag) * b.imag)
                + al.real * a.real
                - (-al.imag) * (-a.imag),
                -(
                    -(-be.imag * b.real + be.real * b.imag)
                    + (-al.imag) * a.real
                    + al.real * (-a.imag)
                ),
            )

            bp = complex(
                al.real * b.real
                - al.imag * b.imag
                + be.real * a.real
                - be.imag * (-a.imag),
                al.real * b.imag
                + al.imag * b.real
                + be.imag * a.real
                + be.real * (-a.imag),
            )

            a = ap
            b = bp

        alpha[ix], beta[ix] = a, b

    return alpha, -beta.conjugate()


def ab2ex(a, b):
    """
    Compute the excitation profile of an RF pulse from Cayley-Klein parameters

    Based on MATLAB code by John Pauly (http://rsl.stanford.edu/research/software.html)

    :param a: Cayley-Klein parameter
    :param b: Cayley-Klein parameter
    :return: transverse magnetisation (mxy)
    :rtype: np.ndarray[complex]
    """

    return 2 * a.conjugate() * b


def b2se(b):
    """
    Compute the spin-echo profile of an RF pulse from Cayley-Klein parameter

    Based on MATLAB code by John Pauly (http://rsl.stanford.edu/research/software.html)

    :param b: Cayley-Klein parameter
    :return: transverse magnetisation (mxy)
    :rtype: np.ndarray[complex]
    """

    return 1j * b * b


def b2fa(b):
    """
    Compute flip-angle profile (in radians) from Cayley-Klein parameter b

    :param b: Cayley-Klein parameter
    :return: flip angle profile (radians)
    :rtype: np.ndarray[float]
    """

    return 2.0 * np.arcsin(np.abs(b))


def rfscale(b1, alpha):
    """
    Scale an RF pulse waveform so that np.sum(b1) = alpha (the flip angle in
    radians)

    :param b1: RF pulse waveform
    :type b1: np.ndarray[complex]
    :param alpha: desired flip angle (radians)
    :type alpha: float
    :return: scaled RF pulse waveform
    :rtype: np.ndarray[complex]
    """
    return b1 * (alpha / np.sum(b1))


def msinc(n, m, a=0, debug=False):
    """
    Compute a windowed sinc pulse of length n, with m-cycles i.e. with a
    time-bandwidth product of 4m

    Based on MATLAB code by John Pauly (http://rsl.stanford.edu/research/software.html)

    :param n: number of points in pulse
    :type n: int
    :param m: number of cycles
    :type m: int
    :param a: apodisation 0=none, 0.46=hamming, 0.5=hanning
    :param debug: show output as plot if true
    :type debug: bool
    :return: windowed sinc pulse
    :rtype: np.ndarray[float]
    """

    x = np.linspace(-n / 2, n / 2, n, endpoint=False) / (n / 2)
    snc = np.sinc(2 * m * x)
    ms = snc * ((1 - a) + a * np.cos(np.pi * x))
    wsnc = ms * 4 * m / n

    if debug:  # pragma: no cover
        plt.plot(wsnc)
        plt.title("msinc output")
        plt.show()

    return wsnc


def rf_sinc(alpha, tbw, a, n, debug=False):
    """
    Create an (apodized)-sinc shaped radio-frequency pulse normalised such
    that sum (rf) = flip angle in radians

    :param alpha: desired flip angle (radians)
    :type alpha: float
    :param tbw: time-bandwidth product
    :type tbw: int
    :param a: apodisation 0=none, 0.46=hamming, 0.5=hanning
    :type a: float
    :param n: number of points in pulse
    :type n: int
    :param debug: show output as plot if true
    :type debug: bool
    :return: windowed scaled sinc pulse
    :rtype: np.ndarray[float]
    """
    s = rfscale(msinc(n, tbw / 4, a, debug), alpha)

    if debug:  # pragma: no cover
        plt.plot(s)
        plt.title("rf_sinc output")
        plt.ylabel(r"$B_1$ (rad)")
        plt.show()

    return s


def gaussian(n, db=-60, debug=False):
    """
    Compute a gaussian pulse of length n.

    b1(t) = exp(-t^2/ 2 * sigma^2)

    Although a gaussian is theoretically non-zero for all t, it falls off
    rapidly for t > sigma. The RF pulse can be terminated when the magnitude
    of the b1 reaches a negligible level. For example at
    t = +/- sigma * sqrt(2 * ln(1000)) i.e. t = +/- 3.717 sigma
    the amplitude is 1/1000 that of the peak (i.e. 60 dB attenuation).
    Therefore pulse duration T = 7.434 * sigma. The FWHM of the Fourier
    transform of the pulse is sqrt(2 * ln(2)) / (pi * sigma).
    Therefore the time bandwidth product of any gaussian pulse is

    (7.434 * sigma) * (0.3748 / sigma) ~ 2.8 i.e. independent of sigma

    :param n: number of points in pulse
    :type n: int
    :param db: decibels of attenuation (should be negative)
    :type db: float
    :param debug: show output as plot if true
    :type debug: bool
    :return: gaussian pulse with TBW ~ 2.8 (if dB = -60)
    :rtype: np.ndarray[float]
    """

    x = np.linspace(-n / 2, n / 2, n, endpoint=False) / (n / 2)
    sigma = np.max(x) / np.sqrt(2 * np.log(10 ** (-db / 20)))
    g = np.exp(-(x**2) / (2 * sigma**2))

    # normalisation so area is ~1 (not exact because of truncation at 60 dB)
    g_norm = g * 2 / (n * sigma * np.sqrt(2 * np.pi))

    if debug:  # pragma: no cover
        plt.plot(g_norm)
        plt.title("gaussian output")
        plt.show()

    return g_norm


def gaussiandb(tbw):
    """
    Compute attenuation of gaussian pulse given time-bandwidth product

    e.g. if db = -60 then the amplitude is 1/1000 at start and end

    :param tbw: pulse time-bandwidth product
    :type tbw: float
    :return: decibels of attenuation (should be negative)
    :rtype: float
    """

    return -20 * np.log10(np.exp(((np.pi * tbw) ** 2) / (16 * np.log(2))))


def rf_gaussian(alpha, tbw, n, debug=False):
    """
    Create an Gaussian shaped radio-frequency pulse normalised such
    that sum (rf) = flip angle in radians

    :param alpha: desired flip angle (radians)
    :type alpha: float
    :param tbw: time-bandwidth product
    :type tbw: int
    :param n: number of points in pulse
    :type n: int
    :param debug: show output as plot if true
    :type debug: bool
    :return: scaled gaussian pulse
    :rtype: np.ndarray[float]
    """

    g = rfscale(gaussian(n, gaussiandb(tbw), debug), alpha)

    if debug:  # pragma: no cover
        plt.plot(g)
        plt.title("rf_gaussian output")
        plt.ylabel(r"$B_1$ (rad)")
        plt.show()

    return g


def spatial_positions(slthick, p=2, n=101):
    """
    Generate an array of n spatial positions between +/-p * slice thickness
    e.g. if slthick = 5 mm, p = 3 and n=101 it would be between +/-15 mm sampled
    at 101 points

    :param slthick: slice thickness (m)
    :type slthick: float
    :param p: multiplier for range of spatial positions
    :type p: float
    :param n: number of spatial positions
    :type n: int
    :return: array of spatial positions
    :rtype: np.ndarray[float]
    """

    return np.linspace(-p * slthick, p * slthick, n, endpoint=False)


def norm_frequencies(z, tbw, slthick):
    """
    Generate an array of normalised frequencies required by abr

    :param z: spatial positions (m)
    :type z: np.ndarray[float]
    :param tbw: time-bandwidth product
    :type tbw: float
    :param slthick: slice thickness (m)
    :type slthick: float
    :return: array of normalised frequencies
    :rtype: np.ndarray[float]
    """

    return (z / slthick) * tbw


def rf2profile(profile_type, rf, tbw, z, slthick, debug=False):
    """
    Calculate profile from RF waveform using the forward SLR transform

    :param profile_type: angle (flip-angle), ex (excitation) or se (spin-echo)
    :type profile_type: str
    :param rf: RF pulse normalised such that sum (rf) = flip angle in radians
    :type rf: np.ndarray[float]
    :param tbw: time-bandwidth product of RF pulse
    :type tbw: int
    :param z: 1-by-Q array of spatial positions (m)
    :type z: np.ndarray[float]
    :param slthick: planned slice thickness of RF pulse (m)
    :type slthick: float
    :param debug: show output as plot if true
    :type debug: bool
    :return: (flip-angle or excitation or spin-echo) profile
    :rtype: np.ndarray[float]
    """

    # x is the normalised frequency axis required by abr
    x = norm_frequencies(z, tbw, slthick)
    a, b = abr(rf, x)

    if profile_type == "angle":
        profile = b2fa(b)
    elif profile_type == "ex":
        profile = ab2ex(a, b)
    elif profile_type == "se":
        profile = b2se(b)

    if debug:  # pragma: no cover
        if profile_type == "angle":
            profile_to_plot = profile
            ylabel = "Flip angle (rad)"
        else:
            profile_to_plot = np.abs(profile)
            ylabel = r"$|m_{xy}|$"

        plt.plot(z, profile_to_plot)
        plt.title("rf2profile output")
        plt.xlabel("Position (m)")
        plt.ylabel(ylabel)
        plt.show()

    return profile
