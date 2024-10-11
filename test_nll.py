import numpy as np
import math
import nll


def test_nll():
    # model = 0, so Ricean PDF simplifies to Rayleigh PDF

    # i.e. P = meas/sigma^2 * exp(-meas^2 / 2 * sigma^2)
    # therefore ln(P) = ln(meas) - 2* ln (sigma) - meas^2 / 2 * sigma ^ 2
    # 2 identical measurements so ln(L) = sum (ln(P)) = 2 * ln (P)
    # So if meas=100 amd sigma =5
    # ln(P) = ln(100) - 2* ln(5) - 100^2 / 2 * 5^2 
    # Therefore -ln(L) = -2 * (ln(100) - 2* ln(5) - 100^2 / 2 * 5^2 ) = 397.2274112777602

    model = np.zeros(2)
    meas = 100 * np.ones(2)
    sigma = 5
    assert math.isclose(nll.nll(model, meas, sigma), 397.2274112777602)


def test_nll_gf():
    # again test model = 0 case by setting A=0 so -ln(L) = 397.2274112777602 as above

    n = 2  # 2 echoes
    meas = 100 * np.ones(n)
    sigma = 5
    assert math.isclose(nll.nll_gf([0, 0.1, 1, sigma], 257, 1, 10E-3, n, np.pi/2, np.pi, meas), 397.2274112777602)
