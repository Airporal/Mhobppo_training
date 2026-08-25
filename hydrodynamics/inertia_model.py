from dataclasses import dataclass
from typing import Tuple
import isaaclab.utils.math as math_utils
import numpy as np
import torch


@dataclass
class InertiaForceModels:
    num_envs: int
    device: torch.device
    dt: float | torch.Tensor
    debug: bool = False

    def calculate_buoyancy_forces(
        self,
        root_quats_w: torch.tensor,  # robot orientations in world frame
        fluid_density: float,  # fluid density
        volumes: torch.tensor,  # rigid body volume
        g_mag: float,  # magnitude of gravity
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute wrenches (forces and torques) due to buoyancy on fully-submerged rigid body in fluid.
        Returned forces and torques are in the body root frame COB with FRD coordinates.
        Note that gravity is applied by Isaac Sim by default.
        """
        buoyancy_directions_w = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        # opposing gravity vector in the world frame
        buoyancy_directions_w[..., 2] = 1.0
        # FRD Body Frame
        buoyancy_directions_b = math_utils.quat_apply_inverse(
            root_quats_w, buoyancy_directions_w
        )
        # print("B: ", buoyancy_directions_b)
        
        buoyancy_forces_b = (
            buoyancy_directions_b * fluid_density * volumes.repeat(1, 3) * g_mag
        )
        # torque = r x F
        buoyancy_torques_b = torch.zeros_like(buoyancy_forces_b)

        if self.debug and False:
            print(f"Buoyancy F: {buoyancy_forces_b}, T: {buoyancy_torques_b}")

        return (buoyancy_forces_b, buoyancy_torques_b)

    def calculate_density_and_viscosity_forces(
        self,
        # [num_envs, 4]
        root_quats_w: torch.tensor,
        # [num_envs, 3]
        root_linvels_b: torch.tensor,
        # [num_envs, 3]
        root_angvels_b: torch.tensor,
        inertias: torch.Tensor,
        water_beta: float,
        fluid_density: float,
        masses: torch.tensor,
        com_positions_b: torch.Tensor,
    ):
        # root_linvels_b = math_utils.quat_apply_inverse(root_quats_w, root_linvels_w)
        # root_angvels_b = math_utils.quat_apply_inverse(root_quats_w, root_angvels_w)
        linvels_at_com_b = root_linvels_b + torch.cross(
            root_angvels_b, com_positions_b, dim=-1
        )
        ri = self._calculate_inferred_half_dimensions(inertias, masses)
        f_d, g_d_com = self.calculate_quadratic_drag_forces(
            linvels_at_com_b, root_angvels_b, ri, fluid_density
        )
        f_v, g_v_com = self.calculate_linear_viscous_forces(
            linvels_at_com_b, root_angvels_b, ri, water_beta
        )
        g_d_cob = torch.cross(com_positions_b, f_d, dim=-1) + g_d_com
        g_v_cob = torch.cross(com_positions_b, f_v, dim=-1) + g_v_com
        hydro_dynamic_F = f_d + f_v
        hydro_dynamic_T = g_d_cob + g_v_cob
        # damping = torch.cat([hydro_dynamic_F, hydro_dynamic_T], dim=-1)
        # viscous_term = torch.cat([f_v, g_v_cob], dim=-1)
        return hydro_dynamic_F, hydro_dynamic_T

    def calculate_quadratic_drag_forces(
        self,
        com_linvels_b: torch.tensor,
        com_angvels_b: torch.tensor,
        ri: torch.tensor,
        fluid_density,
    ):
        rj = torch.roll(ri, 1, 1)
        rk = torch.roll(ri, -1, 1)
        forces = (
            -2.0 * fluid_density * rj * rk * torch.abs(com_linvels_b) * com_linvels_b
        )
        torques = (
            -0.5
            * fluid_density
            * ri
            * (torch.pow(rj, 4) + torch.pow(rk, 4))
            * torch.abs(com_angvels_b)
            * com_angvels_b
        )

        return (forces, torques)

    def calculate_linear_viscous_forces(
        self,
        com_linvels_b: torch.tensor,
        com_angvels_b: torch.tensor,
        ri: torch.tensor,
        fluid_viscosity_beta,
    ):
        r_eq = torch.mean(ri, 1, keepdim=True)
        r_eq = r_eq.repeat(1, 3)
        forces = -6.0 * fluid_viscosity_beta * torch.pi * r_eq * com_linvels_b
        torques = (
            -8.0 * fluid_viscosity_beta * torch.pi * torch.pow(r_eq, 3) * com_angvels_b
        )
        return (forces, torques)

    def _calculate_inferred_half_dimensions(self, inertias, masses):
        """
        Computes inferred half dimensions for an "equivalent inertia box" of the vehicle
        """
        r = torch.sqrt(
            (3 / (2 * masses.repeat(1, 3)))
            * (torch.roll(inertias, 1, 1) + torch.roll(inertias, -1, 1) - inertias)
        )
        return r.to(self.device)

    def reset(self, env_ids) -> None:
        pass

