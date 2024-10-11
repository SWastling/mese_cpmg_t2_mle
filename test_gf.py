import numpy as np
import gf


def test_cpmg_signal_single_point():
    # Does it fall back to single exponential when B1 = 1 and single flip-angle

    A = 1000
    T1 = 10
    T2 = 40e-3
    B1 = 1
    N = 32
    tau = 0.01

    s = gf.GF(A, 256, T1, T2, tau, N, np.pi / 2, np.pi, B1)

    t = np.arange(1, N + 1) * tau
    s_exp = A * np.exp(-t / T2)

    assert np.allclose(s, s_exp)


def test_cpmg_signal_single_point_b1():
    # Check effect of imperfect excitation and refocussing i.e. B1 =0.8

    A = 100
    T1 = 1
    T2 = 0.1
    B1 = 0.8
    N = 16
    tau = 0.01

    s = gf.GF(A, 256, T1, T2, tau, N, np.pi / 2, np.pi, B1)

    s_expected = np.array(
        [
            81.84201706,
            82.45970342,
            68.36677637,
            66.92687342,
            57.73325311,
            54.15259847,
            48.48211515,
            44.30569203,
            40.11229385,
            36.77194582,
            32.831236,
            30.67458059,
            26.89373515,
            25.44913411,
            22.20432261,
            20.96797429,
        ]
    )
    assert np.allclose(s, s_expected)


def test_cpmg_signal_single_point_te0():
    # Does it fall back to single exponential when B1 = 1 and single flip-angle
    # and does it correctly calculate TE=0ms point

    A = 1000
    T1 = 10
    T2 = 40e-3
    B1 = 1
    N = 32
    tau = 0.01

    s = gf.GF(A, 256, T1, T2, tau, N, np.pi / 2, np.pi, B1, True)

    t = np.arange(0, N + 1) * tau
    s_exp = A * np.exp(-t / T2)

    assert np.allclose(s, s_exp)


def test_cpmg_signal_two_points():
    # Does it fall back to single exponential when B1 = 1 and two identical
    # flip-angles

    A = 1000
    T1 = 10
    T2 = 40e-3
    B1 = 1
    N = 32
    tau = 0.01

    fa_ex = np.array([np.pi / 2, np.pi / 2])
    fa_ref = np.array([np.pi, np.pi])

    s = gf.GF(A, 256, T1, T2, tau, N, fa_ex, fa_ref, B1)

    t = np.arange(1, N + 1) * tau
    s_exp = A * np.exp(-t / T2)

    assert np.allclose(s, s_exp)
