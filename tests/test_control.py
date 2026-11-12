"""Least-squares DC voltages and the field they produce at the ion."""

import unittest

import numpy as np

from ion_npe.constants import (
    FORCE_ELECTRODE_VOLTAGES,
    L_SCALE,
    SIM_DT,
    TRANSPORT_SPEED_M_S,
    VOLTAGE_UPDATE_S,
)
from ion_npe import simulate as simulate_mod
from ion_npe.control import electrode_acceleration, solve_transport_voltages
from ion_npe.design import DesignSpace
from ion_npe.dynamics import hold_steps, integrate_acceleration, path_directions
from ion_npe.junction import RF_GAP, build_junction, path_frequencies
from ion_npe.simulate import force_line, simulation_settings
from ion_npe.trajectory import dynamics_grid
from tests.test_junction import _distance_to_rails


class ControlVoltageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        design = DesignSpace().nominal_design()
        sampled = dynamics_grid(design, TRANSPORT_SPEED_M_S, L_SCALE, SIM_DT)
        cls.junction = build_junction(design.junction)
        cls.xyz = sampled["xyz"]
        cls.omega2, _ = path_frequencies(cls.junction, cls.xyz, sampled["tau"])
        cls.directions = path_directions(cls.xyz)
        cls.n_hold = hold_steps(VOLTAGE_UPDATE_S, SIM_DT)
        cls.solved = solve_transport_voltages(
            cls.junction, cls.xyz, cls.omega2, cls.directions, cls.n_hold
        )

    def test_each_update_meets_the_target_well(self):
        solved = self.solved
        self.assertEqual(solved.voltages.shape[1], len(solved.names))
        self.assertEqual(len(solved.voltages), len(solved.update_indices))
        self.assertTrue(np.isfinite(solved.voltages).all())
        self.assertLess(float(np.max(np.abs(solved.voltages))), 10.0)
        self.assertLess(float(solved.residual.max()), 1e-3)
        self.assertLess(float(np.max(np.abs(solved.potential_error))), 1e-6)
        self.assertGreater(float(np.mean(np.abs(solved.voltages) < 1e-3)), 0.5)

    def test_force_at_the_ion_follows_the_held_voltage(self):
        solved = self.solved
        hold = int(solved.update_indices[len(solved.update_indices) // 2])
        center = self.xyz[hold] * L_SCALE
        tangent = self.directions[hold, 0]
        shift = 1e-7
        acc_center = electrode_acceleration(
            self.junction, solved.voltages[hold // self.n_hold], solved.scale, center
        )
        acc_shift = electrode_acceleration(
            self.junction,
            solved.voltages[hold // self.n_hold],
            solved.scale,
            center + shift * tangent,
        )
        measured = np.dot(acc_shift - acc_center, tangent) / shift
        self.assertAlmostEqual(measured, -self.omega2[hold, 0], delta=0.05 * self.omega2[hold, 0])
        self.assertLess(float(np.linalg.norm(acc_center)), 1e4)

    def test_short_transport_stays_finite_without_stray_field(self):
        solved = self.solved
        n_hold = self.n_hold
        stop = min(5 * n_hold, len(self.xyz) - 1)
        r0 = self.xyz[: stop + 1] * L_SCALE

        def acceleration(position, hold):
            slot = min(hold // n_hold, len(solved.voltages) - 1)
            return electrode_acceleration(
                self.junction, solved.voltages[slot], solved.scale, position
            )

        result = integrate_acceleration(
            r0,
            self.omega2[: stop + 1],
            self.directions[: stop + 1],
            n_hold,
            acceleration,
            sim_dt=SIM_DT,
        )
        self.assertTrue(np.isfinite(result["final_quanta"]).all())
        self.assertLess(float(np.linalg.norm(result["final_offset_um"])), 20.0)

    def test_simulation_solves_voltages_on_the_updated_dc(self):
        """A moved RF rail is simulated with the DC that keeps one RF gap."""
        self.assertIs(simulate_mod.build_junction, build_junction)
        space = DesignSpace()
        controls = space.nominal().copy()
        controls[0] = space.upper[0]
        design = space.unpack(controls)
        junction = simulate_mod.build_junction(design.junction)
        center = np.asarray(junction["dc_center"].paths[0], dtype=float)
        distance = _distance_to_rails(center, junction["r"].paths)
        angles = np.arctan2(center[:, 1], center[:, 0])
        facing = np.zeros(len(center), dtype=bool)
        for target in (0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0):
            delta = np.abs(np.arctan2(np.sin(angles - target), np.cos(angles - target)))
            facing |= delta < np.radians(15.0)
        self.assertGreater(int(np.count_nonzero(facing)), 0)
        self.assertLess(float(distance[facing].max()), RF_GAP + 0.01)

        sampled = dynamics_grid(design, TRANSPORT_SPEED_M_S, L_SCALE, SIM_DT)
        n_hold = hold_steps(VOLTAGE_UPDATE_S, SIM_DT)
        stop = min(3 * n_hold, len(sampled["xyz"]) - 1)
        xyz = sampled["xyz"][: stop + 1]
        omega2, _ = path_frequencies(junction, xyz, sampled["tau"][: len(xyz)])
        directions = path_directions(xyz)
        solved = solve_transport_voltages(junction, xyz, omega2, directions, n_hold)
        self.assertEqual(solved.names[0], "dc_center")
        self.assertLess(float(solved.residual.max()), 1e-3)
        acceleration = electrode_acceleration(
            junction, solved.voltages[0], solved.scale, xyz[0] * L_SCALE
        )
        self.assertTrue(np.isfinite(acceleration).all())

    def test_simulation_defaults_to_solved_electrode_voltages(self):
        settings = simulation_settings({"noise_std_Vm": 0.0, "seeds": (0,)})
        self.assertEqual(settings["force"], FORCE_ELECTRODE_VOLTAGES)
        self.assertEqual(force_line(settings), "force: solved DC electrode voltages")
        with self.assertRaisesRegex(ValueError, "solved DC electrode voltages"):
            simulation_settings({"force": "harmonic_well", "noise_std_Vm": 0.0, "seeds": (0,)})


if __name__ == "__main__":
    unittest.main()
