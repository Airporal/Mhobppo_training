'Manage thrust allocation, thruster dynamics, and thrust conversion.'

import numpy as np
import math

from .thruster_dynamics import ThrusterDynamicsMap, Dynamics
from .thruster_conversion import *

from .thruster_configure import (
    ThrusterConfigureMap,
    calculate_thrust_configure_from_usd_vec,
)
from .thruster_conversion import ThrusterConversionMap

try:
    from ..utils import utils
except ImportError:
    import os

    UTILS_PATH = os.path.dirname(os.path.abspath(__file__))
    import sys

    print("Importing utils from:", UTILS_PATH)
    sys.path.append(UTILS_PATH)
    import utils
from isaaclab.utils import configclass
import torch


@configclass
class ThrusterCfg:
    """
    thruster_min & thruster_max : Thruster force limits
    rpm_min & rpm_max : rpm limits
    thruster_configure : thruster configuration matrix, or predefined thruster configuration matrix name
    robot_thruster_number: number of thrusters on the robot
    numEnvs: number of environments
    gain: gain on the input command
    thrust_efficiency: efficiency of converting rotational speed to thrust force,0-1
    propeller_efficiency: efficiency of the propeller rotational speed response,0-1
    dynamic_type: type of thruster dynamics, ZeroOrder, FirstOrder, Yoerger or Bessa
    tau: time constant for first order dynamics
    alpha, beta: parameters for Yoerger dynamics
    Jmsp, Kv1, Kv2, Kt1, Rm: parameters for Bessa dynamics
    conversion_type: type of thruster conversion function, Basic, Bessa, LinearInterp or CubicInterp
    interp_type: type of interpolation, "linear"、"cubic" and others
    inputValues: input values for interpolation, usually the rotational speed
    outputValues: output values for interpolation, usually the thrust force
    rotorConstant: rotorConstant>= 0
    deltaL: deltaL<=0
    deltaR: deltaR>=0
    """

    thruster_min: float = -32.0
    thruster_max: float = 32.0
    wrench_max: list[float] = [
        math.sqrt(2.0) * (thruster_max - thruster_min),
        math.sqrt(2.0) * (thruster_max - thruster_min),
        4 * thruster_max,
        0.0417 * (thruster_max - thruster_min) * 2
        + 0.222 * (thruster_max - thruster_min) * 2,
        0.0417 * (thruster_max - thruster_min) * 2
        - 0.1214 * thruster_min * 2
        + 0.1186 * thruster_max * 2,
        0.1853 * (thruster_max - thruster_min) * 2,
    ]
    wrench_min: list[float] = [
        -math.sqrt(2.0) * (thruster_max - thruster_min),
        -math.sqrt(2.0) * (thruster_max - thruster_min),
        4 * thruster_min,
        -0.0417 * (thruster_max - thruster_min) * 2
        - 0.222 * (thruster_max - thruster_min) * 2,
        -0.0417 * (thruster_max - thruster_min) * 2
        - 0.1214 * thruster_max * 2
        + 0.1186 * thruster_min * 2,
        -0.1853 * (thruster_max - thruster_min) * 2,
    ]
    rpm_min: float = -3278.0
    rpm_max: float = 3278.0
    thruster_configure: torch.Tensor | str = None
    gain: float = 1.0  # to amplify the rpm input command
    thrust_efficiency: float = 1.0
    robot_thruster_number: int = 8
    propeller_efficiency: float = 1.0
    dynamic_type: str = "FirstOrder"
    tau: float = 0.05
    alpha: float = 0.5
    beta: float = 0.5
    Jmsp: float = 0.5
    Kv1: float = 0.5
    Kv2: float = 0.5
    Kt1: float = 0.5
    Rm: float = 0.5
    conversion_type: str = "Basic"
    interp_type: str = "linear"
    rotorConstant: float = 0.1 / 100.0
    rotorConstantL: float = 0.1 / 100.0
    rotorConstantR: float = 0.1 / 100.0
    deltaL: float = 0.0
    deltaR: float = 0.0
    inputValues: list[float] = []
    outputValues: list[float] = []


class Thruster:
    cfg: ThrusterCfg

    def __init__(self, numEnvs: int, cfg: ThrusterCfg, device: torch.device):
        Thruster.cfg = cfg
        self.thrusterDynamics: Dynamics = None
        self.conversionFunction: ConversionFunction = None
        self._numEnvs = numEnvs
        self._device = device
        # n,t_num
        self._inputCommand = torch.zeros(
            (self._numEnvs, self.cfg.robot_thruster_number),
            dtype=torch.float32,
            device=self._device,
            requires_grad=False,
        )
        # n,6
        self._wrench_min = torch.tensor(
            self.cfg.wrench_min, dtype=torch.float32, device=self._device
        )
        # n,6
        self._wrench_max = torch.tensor(
            self.cfg.wrench_max, dtype=torch.float32, device=self._device
        )
        self._wrench_scale = 0.5 * (self._wrench_max - self._wrench_min)
        self._wrench_bias = 0.5 * (self._wrench_max + self._wrench_min)
        self._isOn = True
        self._load_configure()

    def _load_configure(self, debug=True):
        """

            BLUEROV_THRUSTER_CONFIG = np.array(
            [
                [0.707, 0.707, -0.707, -0.707, 0, 0, 0, 0],
                [-0.707, 0.707, -0.707, 0.707, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 1, 1, -1],
                [0.06, -0.06, 0.06, -0.06, -0.218, -0.218, 0.218, 0.218],
                [0.06, 0.06, -0.06, -0.06, 0.120, -0.120, 0.120, -0.120],
                [-0.1888, 0.1888, 0.1888, -0.1888, 0, 0, 0, 0],
            ]
        )
        """

        
        if isinstance(self.cfg.thruster_configure, str):
            
            self.thruster_configure = ThrusterConfigureMap[self.cfg.thruster_configure]
            if debug:
                print(
                    "Loaded thruster configuration:",
                    self.cfg.thruster_configure,
                    "\n",
                    self.thruster_configure,
                )
        elif isinstance(self.cfg.thruster_configure, torch.Tensor):
            
            self.thruster_configure = self.cfg.thruster_configure.clone()
        
        thrusterDynamics = ThrusterDynamicsMap[self.cfg.dynamic_type]
        if self.cfg.dynamic_type == "FirstOrder":
            self.thrusterDynamics = thrusterDynamics(
                self._numEnvs,
                self.cfg.robot_thruster_number,
                self.cfg.tau,
                self._device,
            )
        elif self.cfg.dynamic_type == "Yoerger":
            self.thrusterDynamics = thrusterDynamics(
                self._numEnvs,
                self.cfg.robot_thruster_number,
                self.cfg.alpha,
                self.cfg.beta,
                self._device,
            )
        elif self.cfg.dynamic_type == "Bessa":
            self.thrusterDynamics = thrusterDynamics(
                self._numEnvs,
                self.cfg.robot_thruster_number,
                self.cfg.Jmsp,
                self.cfg.Kv1,
                self.cfg.Kv2,
                self.cfg.Kt1,
                self.cfg.Rm,
                self._device,
            )
        else:
            # zero order dynamics
            self.thrusterDynamics = thrusterDynamics(
                self._numEnvs,
                self.cfg.robot_thruster_number,
                self._device,
            )
        
        conversionFunction = ThrusterConversionMap[self.cfg.conversion_type]
        if self.cfg.conversion_type == "Bessa":
            self.conversionFunction = conversionFunction(
                self.cfg.rotorConstantL,
                self.cfg.rotorConstantR,
                self.cfg.deltaL,
                self.cfg.deltaR,
            )
        elif self.cfg.conversion_type == "Interp":
            self.conversionFunction = conversionFunction(
                self.cfg.inputValues, self.cfg.outputValues, self.cfg.interp_type
            )
        else:
            # basic conversion function
            self.conversionFunction = conversionFunction(self.cfg.rotorConstant)

    def reset_all(self):
        self.thrusterDynamics.reset_all()

    def reset(self, maskArr):
        self.thrusterDynamics.reset(maskArr)

    def update(
        self, cmd: torch.tensor, dt: torch.tensor, type: str = "pwm", debug=True
    ):
        """
        cmd: n,t_num, input command for each thruster in each env,[-1,1]
        dt: sim physics time interval between last update and current update
        type: "pwm" or "wrench", input command type
        return:
            self._thrustForce: n,t_num, thrust force generated by each thruster in each env
                                if >0, thrustForce is inside
            wrench_b: n,6, wrench in FRD body frame generated by all thrusters in each env

        """
        if type == "pwm":
            self._inputCommand = cmd.clone()
            pwm, desire_rpm = utils.cmd_to_pwm_to_rpm(self._inputCommand)
            
        elif type == "wrench":
            cmd = cmd * self._wrench_scale + self._wrench_bias
            desire_thrust = utils.wrench_to_thrust(
                cmd, self.thruster_configure.to(self._device)
            )
            desire_rpm = self.conversionFunction.inv_convert(desire_thrust)
        if debug:
            print("Cmd:", cmd)
            print("PWM:", pwm)
            print("Desired RPM:", desire_rpm)
        if self._isOn:
            dynamicsInput = desire_rpm.clone()
        else:
            dynamicsInput = torch.zeros_like(desire_rpm)
        # n,n_thrusters rotation speed of each propeller in each env
        if debug:
            print("Dynamics Input RPM:", dynamicsInput)
        dynamicState = self.cfg.propeller_efficiency * self.thrusterDynamics.update(
            dynamicsInput, dt
        )
        # n,n_thrusters thrust force generated by each propeller in each env

        dynamicState = (
            torch.clamp(dynamicState, self.cfg.rpm_min, self.cfg.rpm_max)
            * self.cfg.gain
        )
        if debug:
            print("Dynamic State RPM:", dynamicState)
        self._thrustForce = (
            self.cfg.thrust_efficiency * self.conversionFunction.convert(dynamicState)
        )
        if debug:
            print("Thrust Force:", self._thrustForce)
        # Use the thrust force limits
        self._thrustForce = torch.clamp(
            self._thrustForce, self.cfg.thruster_min, self.cfg.thruster_max
        )
        # n,6 wrench in body frame generated by all thrusters in each env
        if debug:
            print("Clamped Thrust Force:", self._thrustForce[:3, :])
        wrench_b = utils.thrust_to_wrench(
            self._thrustForce, self.thruster_configure.to(self._device)
        )
        # thrusForce >0 means thrust is inside in physical world, but for set_external_force_and_torque API,it should be opposite.
        # wrench_b is defined in FRD body frame.
        return -self._thrustForce, wrench_b

    def getCmd(self):
        return self._inputCommand.clone()

    def getVelocity(self):
        return self.thrusterDynamics.state.clone()
