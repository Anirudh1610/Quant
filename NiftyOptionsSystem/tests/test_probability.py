from __future__ import annotations

import numpy as np
import pytest

from core.probability import probability_of_touch

S, T, SIGMA = 24300.0, 6 / 365, 0.13


class TestEdgeCases:
    def test_barrier_equals_spot_is_certain(self):
        assert probability_of_touch(S, S, T, SIGMA) == pytest.approx(1.0)

    def test_zero_time_nontrivial_barrier_is_zero(self):
        assert probability_of_touch(S, S * 1.05, 0.0, SIGMA) == 0.0

    def test_zero_time_at_barrier_is_certain(self):
        assert probability_of_touch(S, S, 0.0, SIGMA) == 1.0

    def test_zero_vol_nontrivial_barrier_is_zero(self):
        assert probability_of_touch(S, S * 1.05, T, 0.0) == 0.0

    def test_bounded_0_1_across_wide_input_range(self):
        for barrier_mult in (0.7, 0.85, 0.95, 0.99, 1.0, 1.01, 1.05, 1.2, 1.5):
            for sigma in (0.05, 0.13, 0.30, 0.60):
                for t in (1 / 365, 30 / 365, 1.0):
                    p = probability_of_touch(S, S * barrier_mult, t, sigma)
                    assert 0.0 <= p <= 1.0


class TestMonotonicity:
    def test_farther_barrier_has_lower_touch_probability(self):
        near = probability_of_touch(S, S * 1.01, T, SIGMA)
        far = probability_of_touch(S, S * 1.10, T, SIGMA)
        assert near > far

    def test_more_time_increases_touch_probability(self):
        short_t = probability_of_touch(S, S * 1.05, 5 / 365, SIGMA)
        long_t = probability_of_touch(S, S * 1.05, 60 / 365, SIGMA)
        assert long_t > short_t

    def test_higher_vol_increases_touch_probability(self):
        low_vol = probability_of_touch(S, S * 1.05, T, 0.10)
        high_vol = probability_of_touch(S, S * 1.05, T, 0.40)
        assert high_vol > low_vol


class TestDriftlessSymmetry:
    def test_symmetric_log_barriers_equal_probability_at_true_zero_drift(self):
        # nu = mu - 0.5*sigma^2 must be exactly 0 for up/down touch symmetry
        # to hold on symmetric log-distance barriers -- i.e. mu = 0.5*sigma^2,
        # NOT mu=0 (mu=0 gives a slightly negative nu, favoring the downside
        # touch, which is deliberately the module's default since it matches
        # a driftless *forward*, not a driftless log-price).
        mu_for_zero_nu = 0.5 * SIGMA ** 2
        log_dist = 0.04
        upper = S * np.exp(log_dist)
        lower = S * np.exp(-log_dist)
        p_up = probability_of_touch(S, upper, T, SIGMA, mu=mu_for_zero_nu)
        p_down = probability_of_touch(S, lower, T, SIGMA, mu=mu_for_zero_nu)
        assert p_up == pytest.approx(p_down, abs=1e-10)

    def test_default_zero_mu_slightly_favors_downside_touch(self):
        # with mu=0 (default), nu = -0.5*sigma^2 < 0, so the downward barrier
        # is (very slightly) more likely to be touched than the symmetric
        # upward one -- documented consequence of the module's drift choice.
        log_dist = 0.04
        upper = S * np.exp(log_dist)
        lower = S * np.exp(-log_dist)
        p_up = probability_of_touch(S, upper, T, SIGMA)
        p_down = probability_of_touch(S, lower, T, SIGMA)
        assert p_down > p_up


class TestAgainstMonteCarlo:
    @pytest.mark.parametrize("barrier_mult,mu", [(1.05, 0.0), (0.93, 0.0), (1.08, 0.10)])
    def test_matches_simulated_gbm_touch_frequency(self, barrier_mult, mu):
        rng = np.random.default_rng(123)
        n_paths, n_steps = 20_000, 200
        dt = T / n_steps
        barrier = S * barrier_mult

        log_paths = np.full(n_paths, np.log(S))
        touched = np.zeros(n_paths, dtype=bool)
        log_barrier = np.log(barrier)
        drift = (mu - 0.5 * SIGMA ** 2) * dt
        vol = SIGMA * np.sqrt(dt)

        for _ in range(n_steps):
            log_paths = log_paths + drift + vol * rng.standard_normal(n_paths)
            if barrier >= S:
                touched |= log_paths >= log_barrier
            else:
                touched |= log_paths <= log_barrier

        mc_prob = touched.mean()
        closed_form = probability_of_touch(S, barrier, T, SIGMA, mu=mu)

        # generous tolerance -- this is a statistical check, not exact
        se = np.sqrt(mc_prob * (1 - mc_prob) / n_paths)
        assert abs(mc_prob - closed_form) < 6 * se + 0.01
