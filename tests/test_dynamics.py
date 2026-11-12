"""Integrator, path frame, and the stray-field window."""

import unittest

import numpy as np

from ion_npe.constants import (
    HBAR,
    HEATING_QUANTA_PER_S,
    L_SCALE,
    MASS,
    NOISE_SEEDS,
    NOISE_STD_VM,
    OMEGA_AX,
    OMEGA_RAD,
    SIM_DT,
    VOLTAGE_UPDATE_S,
)
from ion_npe.dynamics import (
    heat_velocity,
    heating_rate,
    hold_steps,
    integrate,
    path_directions,
    path_frame_noise,
)
from ion_npe.simulate import simulation_settings


class DynamicsTests(unittest.TestCase):
    def test_voltage_hold_matches_one_hundred_nanoseconds(self):
        steps = hold_steps(VOLTAGE_UPDATE_S)
        self.assertLessEqual(abs(steps * SIM_DT - 100e-9), 0.5 * SIM_DT)

    def test_a_stray_field_draws_enough_samples_for_the_stochastic_network(self):
        settings = simulation_settings()
        self.assertEqual(settings["noise_std_Vm"], NOISE_STD_VM)
        self.assertEqual(settings["noise_std_Vm"], 100.0)
        self.assertEqual(settings["seeds"], NOISE_SEEDS)
        self.assertGreaterEqual(len(settings["seeds"]), 10)
        with self.assertRaises(ValueError):
            simulation_settings({"noise_std_Vm": 100.0, "seeds": (0, 1, 2)})

    def test_a_quiet_run_can_use_one_sample(self):
        settings = simulation_settings({"noise_std_Vm": 0.0, "seeds": (0,)})
        self.assertEqual(settings["seeds"], (0,))
        quiet = simulation_settings({"noise_std_Vm": 0.0})
        self.assertEqual(quiet["seeds"], (0,))
        self.assertEqual(quiet["heating"], "off")
        with self.assertRaises(ValueError):
            simulation_settings({"heating": "both"})

    def test_heating_is_300_per_second_at_1_5_mhz(self):
        omega = 2.0 * np.pi * 1.5e6
        rate = heating_rate(np.array([omega ** 2, (2.0 * omega) ** 2]))
        self.assertAlmostEqual(float(rate[0]), HEATING_QUANTA_PER_S, places=6)
        self.assertAlmostEqual(float(rate[1]), HEATING_QUANTA_PER_S / 4.0, places=6)

    def test_a_velocity_kick_adds_one_quantum_of_energy(self):
        omega = 2.0 * np.pi * 1.5e6
        omega2 = np.full(3, omega ** 2)
        direction = np.eye(3)
        vel = np.array([0.02, -0.01, 0.0])
        dt = 1e-6
        updated = heat_velocity(vel, np.zeros(3), direction, omega2, dt)
        gained = 0.5 * MASS * (updated ** 2 - vel ** 2)
        expected = heating_rate(omega2) * dt * HBAR * omega
        np.testing.assert_allclose(gained, expected, rtol=1e-10)

    def test_end_heating_is_added_only_to_the_final_quanta(self):
        n = 5
        r0 = np.zeros((n, 3))
        r0[:, 0] = np.linspace(0.0, 1e-6, n)
        omega = 2.0 * np.pi * 1.0e6
        omega2 = np.full((n, 3), omega ** 2)
        directions = np.repeat(np.eye(3)[None, :, :], n, axis=0)
        quiet = integrate(r0, omega2, directions, n_hold=1, heating="off")
        booked = integrate(r0, omega2, directions, n_hold=1, heating="end")
        kicked = integrate(r0, omega2, directions, n_hold=1, heating="velocity")
        added = heating_rate(omega2[0]) * SIM_DT * (n - 1)
        np.testing.assert_allclose(booked["final_quanta"], quiet["final_quanta"] + added, rtol=1e-8)
        np.testing.assert_allclose(booked["peak_quanta"], quiet["peak_quanta"], rtol=1e-8)
        self.assertGreater(float(kicked["final_quanta"].sum()), float(quiet["final_quanta"].sum()))

    def test_noise_and_its_slope_vanish_at_the_ends(self):
        field = path_frame_noise(2001, 100.0, seed=3)
        np.testing.assert_allclose(field[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(field[-1], 0.0, atol=1e-12)
        edge = np.max(np.abs(field[1] - field[0]))
        interior = np.max(np.abs(np.diff(field, axis=0)))
        self.assertLess(edge, 0.05 * interior)

    def test_flat_path_uses_the_horizontal_frame(self):
        tau = np.linspace(0.0, 1.0, 80)
        xy = np.column_stack([
            -2.0 + 1.5 * tau + 0.2 * np.sin(2.0 * np.pi * tau),
            0.15 * tau ** 2,
        ])
        xyz = np.column_stack([xy, np.full(len(tau), 0.9)])
        frame = path_directions(xyz)
        deriv = np.gradient(xy, axis=0)
        tangent = deriv / np.linalg.norm(deriv, axis=1, keepdims=True)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        np.testing.assert_allclose(frame[:, 0, :2], tangent, atol=1e-8)
        np.testing.assert_allclose(frame[:, 0, 2], 0.0, atol=1e-8)
        np.testing.assert_allclose(frame[:, 1, :2], normal, atol=1e-8)
        np.testing.assert_allclose(frame[:, 2, :], np.broadcast_to([0.0, 0.0, 1.0], (len(frame), 3)), atol=1e-8)
        for row in frame[::10]:
            np.testing.assert_allclose(row @ row.T, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(np.linalg.det(row), 1.0, places=6)

    def test_static_well_holds_quanta(self):
        n = 8000
        r0 = np.zeros((n, 3))
        omega2 = np.tile([OMEGA_AX ** 2, OMEGA_RAD ** 2, OMEGA_RAD ** 2], (n, 1))
        directions = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        rested = integrate(r0, omega2, directions, n_hold=1)
        self.assertTrue(np.all(rested["final_quanta"] < 1e-8))

        shift = 1e-6
        moved = integrate(r0, omega2, directions, n_hold=1, r_init=[shift, 0.0, 0.0])
        expected = 0.5 * MASS * OMEGA_AX * shift ** 2 / HBAR
        self.assertAlmostEqual(moved["final_quanta"][0], expected, delta=1e-3 * expected)
        self.assertLess(moved["final_quanta"][1], 1e-6 * expected)
        self.assertAlmostEqual(moved["peak_quanta"][0], expected, delta=1e-2 * expected)
        self.assertGreater(L_SCALE, 0.0)


if __name__ == "__main__":
    unittest.main()
