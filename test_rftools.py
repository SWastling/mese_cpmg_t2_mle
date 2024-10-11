import itertools
import math
import numpy as np
import rftools


def test_abr_1():
    # simple 90 degree hard pulse
    rf = np.zeros(5)
    rf = np.append(rf, np.ones(5) * (1 / 5) * np.pi / 2)
    rf = np.append(rf, np.zeros(5))
    x = np.zeros(15)
    assert np.sum(rf) == np.pi / 2

    a, b = rftools.abr(rf, x)
    assert np.allclose(a, (np.ones(15) / np.sqrt(2)))
    assert np.allclose(b, (1j * np.ones(15) / np.sqrt(2)))
    assert np.allclose(np.abs(a) ** 2 + np.abs(b) ** 2, np.ones(15))


def test_ab2ex():
    a = np.ones(5) / np.sqrt(2)
    b = 1j * np.ones(5) / np.sqrt(2)
    mxy = rftools.ab2ex(a, b)
    assert np.allclose(mxy, 2 * a.conjugate() * b)


def test_b2se():
    b = 1j * np.ones(5)
    mxy = rftools.b2se(b)
    assert np.allclose(mxy, -1j * np.ones(5))


def test_b2fa():
    b = 1j * np.ones(5)
    fa = rftools.b2fa(b)
    assert np.allclose(fa, np.ones(5) * np.pi)


def test_rfscale_1():
    b1 = rftools.rfscale(rftools.msinc(1000, 4), np.pi / 2)
    assert np.sum(b1) == np.pi / 2


def test_rfscale_2():
    b1 = rftools.rfscale(rftools.msinc(1000, 4), np.pi)
    assert np.sum(b1) == np.pi


def test_msinc_1():
    n = 1000
    m = 4
    b1 = rftools.msinc(n, m)

    assert len(b1) == n
    assert math.isclose(np.sum(b1), 0.9747537836714414)
    # check number of zeros is 4m
    assert len(list(itertools.groupby(b1, lambda b1: b1 > 0))) + 1 == 4 * m


def test_msinc_2():
    n = 1000
    m = 4
    alpha = 0.46
    b1 = rftools.msinc(n, m, alpha)

    assert len(b1) == n
    assert math.isclose(np.sum(b1), 0.9981618186318053)
    # check number of zeros is 4m
    assert len(list(itertools.groupby(b1, lambda b1: b1 > 0))) + 1 == 4 * m


def test_msinc_3():
    n = 1000
    m = 4
    alpha = 0.5
    b1 = rftools.msinc(n, m, alpha)

    assert len(b1) == n
    assert math.isclose(np.sum(b1), 1.0001972999327067)
    # check number of zeros is 4m
    assert len(list(itertools.groupby(b1, lambda b1: b1 > 0))) + 1 == 4 * m


def test_rf_sinc():
    alpha = np.pi / 2
    tbw = 4
    a = 0.46
    n = 1000

    b1 = rftools.rf_sinc(alpha, tbw, a, n)

    assert len(b1) == n
    assert math.isclose(np.sum(b1), np.pi / 2)
    # check number of zeros is 4m
    assert len(list(itertools.groupby(b1, lambda b1: b1 > 0))) + 1 == tbw


def test_gaussian_1():
    n = 1000
    b1 = rftools.gaussian(n)

    assert len(b1) == n
    assert math.isclose(np.sum(b1), 0.999804183780233)
    assert (b1[0] / np.max(b1)) < 0.0011
    assert (b1[-1] / np.max(b1)) < 0.0011


def test_gaussian_2():
    n = 1000
    b1 = rftools.gaussian(n, -40)

    assert len(b1) == n
    assert math.isclose(np.sum(b1), 0.9976414880488982)
    assert (b1[0] / np.max(b1)) < 0.011
    assert (b1[-1] / np.max(b1)) < 0.011


def test_gaussiandb():
    assert math.isclose(rftools.gaussiandb(2.786065490101816), -60)


def test_rf_gaussian():
    alpha = np.pi / 2
    tbw = 4
    n = 1000

    b1 = rftools.rf_gaussian(alpha, tbw, n)

    assert len(b1) == n
    assert math.isclose(np.sum(b1), np.pi / 2)


def test_spatial_positions():
    n = 100
    slthick = 2
    p = 2

    x = rftools.spatial_positions(slthick, p, n)
    assert len(x) == n
    assert x[0] == -4
    assert x[1] == -3.92
    assert x[-1] == 3.92


def test_norm_frequencies():
    n = 100
    slthick = 2
    p = 2

    z = rftools.spatial_positions(slthick, p, n)
    tbw = 4

    f = rftools.norm_frequencies(z, tbw, slthick)
    assert len(f) == n
    assert f[0] == -8
    assert f[1] == -7.84
    assert f[-1] == 7.84


def test_rf2profile_1():
    # simple 90 degree hard pulse
    rf = np.ones(5) * (1 / 5) * np.pi / 2

    slthick = 1
    tbw = 0
    z = rftools.spatial_positions(slthick, 2, 5)

    profile = rftools.rf2profile("angle", rf, tbw, z, slthick)
    assert np.allclose(profile, np.ones(5) * np.pi / 2)


def test_rf2profile_2():
    # simple 90 degree hard pulse
    rf = np.ones(5) * (1 / 5) * np.pi / 2

    slthick = 1
    tbw = 0
    z = rftools.spatial_positions(slthick, 2, 5)

    profile = rftools.rf2profile("ex", rf, tbw, z, slthick)
    assert np.allclose(profile, np.array([0 + 1j, 0 + 1j, 0 + 1j, 0 + 1j, 0 + 1j]))


def test_rf2profile_3():
    # simple 90 degree hard pulse
    rf = np.ones(5) * (1 / 5) * np.pi / 2

    slthick = 1
    tbw = 0
    z = rftools.spatial_positions(slthick, 2, 5)

    profile = rftools.rf2profile("se", rf, tbw, z, slthick)
    assert np.allclose(
        profile, np.array([0 - 0.5j, 0 - 0.5j, 0 - 0.5j, 0 - 0.5j, 0 - 0.5j])
    )
