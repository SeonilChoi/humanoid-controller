#!/usr/bin/env python3
"""BAM (Rhoban "Better Actuator Models") actuator models for DAMIAO MIT-mode motors + CLI wrapper.

Registers ``dm_j8006``, ``dm_j6006`` and ``dm_j4340`` in ``bam.actuators.actuators`` and forwards to the BAM
command-line tools, so the stock BAM pipeline runs on the logs of example_damiao_bam:

    python damiao_bam.py process --raw <session>/bam --logdir processed/dm_j8006 --dt 0.002
    python damiao_bam.py fit  --actuator dm_j8006 --model m2 --logdir processed/dm_j8006 \
                              --output params/dm_j8006/m2.json --trials 2000 --validation_kp 20
    python damiao_bam.py plot --actuator dm_j8006 --logdir processed/dm_j8006 --params params/dm_j8006/m2.json --sim

Needs BAM (Python 3.12): pip install better-actuator-models   (or: pip install -e <bam clone>)

Model (the firmware MIT law, as BAM's UnitreeGo1Actuator):
    tau = ratio * kp * (q_target - q) + kd * (0 - dq),   v_des = 0 and tau_ff = 0 in the BAM logs
    tau clipped to +-max_torque, and to the linear torque / speed line max_torque * (1 -+ dq / max_velocity)
The logs are NO LOAD (mass = arm_mass = length = 0): the inertia is the armature alone. Friction models: m1 / m2
(constant + viscous [+ Stribeck]) are fully identifiable; m3 / m4 load-dependent friction only through the motor's
own torque; m5 / m6 (directional, external load) are not identifiable without a load.
"""

from __future__ import annotations

import runpy
import sys

try:
    from bam.actuator import Actuator
    from bam.parameter import Parameter
    from bam.testbench import Pendulum
    import bam.actuators
except ImportError as e:  # pragma: no cover
    sys.exit(f"BAM not installed ({e}). pip install better-actuator-models  (Python 3.12)")


class DamiaoActuator(Actuator):
    """DAMIAO DM-J series in MIT mode (torque out = kp * e + kd * de + ff, computed in the motor)."""

    def __init__(self, testbench_class, max_torque: float, max_velocity: float, armature: float,
                 max_friction: float = 2.0, max_viscous: float = 1.0):
        super().__init__(testbench_class)
        self.max_torque_init = max_torque
        self.max_velocity_init = max_velocity
        self.armature_init = armature
        self.max_friction = max_friction
        self.max_viscous = max_viscous
        self.kp = 10.0
        self.kd = 0.3

    def initialize(self):
        # Commanded -> real torque gain (torque constant / current loop error)
        self.model.ratio = Parameter(1.0, 0.5, 1.5)
        # Peak torque [Nm]
        self.model.max_torque = Parameter(self.max_torque_init, 0.5 * self.max_torque_init, 1.5 * self.max_torque_init)
        # No-load speed of the torque / speed line [rad/s]
        self.model.max_velocity = Parameter(self.max_velocity_init, 0.5 * self.max_velocity_init,
                                            3.0 * self.max_velocity_init)
        # Output-side apparent inertia (rotor * ratio^2 + gearbox + output) [kg m^2]
        self.model.armature = Parameter(self.armature_init, 1e-5, 50.0 * self.armature_init)

        self.model.max_friction_base = self.max_friction
        self.model.max_load_friction = 1.0
        self.model.max_viscous_friction = self.max_viscous

    def load_log(self, log: dict):
        super().load_log(log)
        self.kp = log["kp"]
        self.kd = log["damping"]

    def control_unit(self) -> str:
        return "N.m"

    def compute_control(self, q_target, q, dq, dt):
        torque = (q_target - q) * self.kp * self.model.ratio.value + self.kd * (0.0 - dq)
        return self.backend.clamp(torque, -self.model.max_torque.value, self.model.max_torque.value)

    def compute_torque(self, control, torque_enable, q, dq):
        t_max = self.model.max_torque.value
        v_max = self.model.max_velocity.value
        upper = self.backend.clamp(t_max * (1.0 - dq / v_max), -t_max, t_max)
        lower = self.backend.clamp(-t_max * (1.0 + dq / v_max), -t_max, t_max)
        return self.backend.clamp(control * torque_enable, lower, upper)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value


# Initial guesses only (the fit moves them). Peak torque / no-load speed from the DAMIAO datasheets at 24 V
# (J8006 / J6006 6:1, J4340 40:1); armature = output-side inertia guess.
MOTORS = {
    "dm_j8006": dict(max_torque=20.0, max_velocity=20.0, armature=0.004, max_friction=2.0, max_viscous=1.0),
    "dm_j6006": dict(max_torque=12.0, max_velocity=30.0, armature=0.002, max_friction=1.5, max_viscous=1.0),
    "dm_j4340": dict(max_torque=27.0, max_velocity=5.5, armature=0.05, max_friction=5.0, max_viscous=5.0),
}

for _name, _kw in MOTORS.items():
    bam.actuators.actuators[_name] = (lambda kw=_kw: DamiaoActuator(Pendulum, **kw))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("process", "fit", "plot"):
        print(__doc__)
        sys.exit(1)
    command = sys.argv[1]
    sys.argv = [f"bam.{command}"] + sys.argv[2:]
    runpy.run_module(f"bam.{command}", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
