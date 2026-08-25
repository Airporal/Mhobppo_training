"""
This script contains various thruster dynamic models.
These models transform the input command signal into actual thrust rotation velocities.
Import command should be desired rotation speed of the propeller.
Import command range: -4000rpm to 4000rpm
"""

import torch


class Dynamics:
    def __init__(self, numEnvs: int, robot_thruster_number: int, device: torch.device):
        self.numEnvs = numEnvs
        self.robot_thruster_number = robot_thruster_number
        self.device = device
        self.reset_all()

    def update(self, desired_rpm: torch.tensor, dt: torch.tensor) -> torch.tensor:
        """
        desired_rpm: n,n_thrusters
        dt: sim physics time interval between last update and current update
        return: n,n_thrusters rotation speed of each propeller in each env
        """
        raise NotImplementedError()

    def reset(self, maskArr):
        'Reset thruster state for completed environments.'
        self.state[maskArr, :] = 0.0

    def reset_all(self):
        'Reset thruster state for every environment.'
        self.state = torch.zeros(
            (self.numEnvs, self.robot_thruster_number),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

    def get_state(self):
        return self.state.clone()


class DynamicsZeroOrder(Dynamics):
    """
    Zero order thruster dynamics: output thrust equals input command
    """

    def __init__(self, numEnvs: int, robot_thruster_number: int, device: torch.device):
        super().__init__(
            numEnvs=numEnvs, robot_thruster_number=robot_thruster_number, device=device
        )

    def update(self, desired_rpm: torch.tensor, dt: torch.tensor):
        # desired_rpm = cmd_to_pwm_to_rpm(cmd)
        self.state[:] = desired_rpm
        return self.state  # n,n_thrusters


class DynamicsFirstOrder(Dynamics):
    """
    First order thruster dynamics: dT/dt = (T_cmd - T)/tau
    """

    def __init__(
        self, numEnvs: int, robot_thruster_number: int, tau: float, device: torch.device
    ):
        super().__init__(
            numEnvs=numEnvs, robot_thruster_number=robot_thruster_number, device=device
        )
        self.tau = tau

    def update(self, desired_rpm: torch.tensor, dt: torch.tensor):
        """
        desired_rpm: n,n_thrusters
        dt: sim physics time interval between last update and current update
        """
        # desired_rpm = cmd_to_pwm_to_rpm(cmd)
        # print("dt: ", dt, " tau: ", self.tau)
        alpha = torch.exp(-dt / self.tau)  # e^(-1/5)
        self.state[:] = (
            self.state * alpha.unsqueeze(-1) + (1.0 - alpha).unsqueeze(-1) * desired_rpm
        )
        return self.state  # n,n_thrusters


class ThrusterDynamicsYoerger(Dynamics):
    """
    state += dt * (beta * desired_rpm - alpha * state * abs(state))
    """

    def __init__(
        self,
        numEnvs: int,
        robot_thruster_number: int,
        alpha: float,
        beta: float,
        device: torch.device,
    ):
        super().__init__(
            numEnvs=numEnvs, robot_thruster_number=robot_thruster_number, device=device
        )
        self.alpha = torch.tensor(alpha, device=device, requires_grad=False).resize(1)
        self.beta = torch.tensor(beta, device=device, requires_grad=False).resize(1)

    def update(self, desired_rpm: torch.tensor, dt: torch.tensor):
        # desired_rpm = cmd_to_pwm_to_rpm(cmd)
        self.state += dt * (
            self.beta * desired_rpm - self.alpha * self.state * torch.abs(self.state)
        )

        return self.state  # n,n_thrusters


class ThrusterDynamicsBessa(Dynamics):
    def __init__(
        self,
        numEnvs: int,
        robot_thruster_number: int,
        Jmsp: float,
        Kv1: float,
        Kv2: float,
        Kt: float,
        Rm: float,
        device: torch.device,
    ):
        super().__init__(
            numEnvs=numEnvs, robot_thruster_number=robot_thruster_number, device=device
        )

        self.Jmsp = torch.tensor(Jmsp, device=device, requires_grad=False).resize(1)
        self.Kv1 = torch.tensor(Kv1, device=device, requires_grad=False).resize(1)
        self.Kv2 = torch.tensor(Kv2, device=device, requires_grad=False).resize(1)
        self.Kt = torch.tensor(Kt, device=device, requires_grad=False).resize(1)
        self.Rm = torch.tensor(Rm, device=device, requires_grad=False).resize(1)

    def update(self, desired_rpm: torch.tensor, dt: torch.tensor):
        # desired_rpm = cmd_to_pwm_to_rpm(cmd)
        self.state += (
            dt
            * (
                desired_rpm * self.Kt / self.Rm
                - self.Kv1 * self.state
                - self.Kv2 * self.state * torch.abs(self.state)
            )
            / self.Jmsp
        )

        return self.state  # n,n_thrusters


ThrusterDynamicsMap = {
    "ZeroOrder": DynamicsZeroOrder,
    "FirstOrder": DynamicsFirstOrder,
    "Yoerger": ThrusterDynamicsYoerger,
    "Bessa": ThrusterDynamicsBessa,
}
