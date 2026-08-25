'Test the Fossen dynamics implementation.'
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


@dataclass
class FossenForceModels:
    num_envs: int
    device: torch.device
    dt: float | torch.Tensor
    debug: bool = False

    def __post_init__(self):
        self.added_mass = torch.tensor(
            [5.5, 12.7, 14.57, 0.12, 0.12, 0.12],
            device=self.device,
            dtype=torch.float32,
        )
        self.linear_damping = torch.tensor(
            [4.03, 6.22, 5.18, 0.07, 0.07, 0.07],
            device=self.device,
            dtype=torch.float32,
        )
        self.quadratic_damping = torch.tensor(
            [18.18, 21.66, 36.99, 1.55, 1.55, 1.55],
            device=self.device,
            dtype=torch.float32,
        )
        self._prev_nu = torch.zeros(
            self.num_envs, 6, device=self.device, dtype=torch.float32
        )
        self._reset_mask = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._has_reset = False

    def _coriolis_added(self, nu: torch.Tensor) -> torch.Tensor:
        m = self.added_mass
        u, v, w, p, q, r = [nu[:, i] for i in range(6)]
        c = torch.zeros((self.num_envs, 6, 6), device=self.device, dtype=nu.dtype)
        c[:, 0, 4] = m[2] * w
        c[:, 0, 5] = -m[1] * v
        c[:, 1, 3] = -m[2] * w
        c[:, 1, 5] = m[0] * u
        c[:, 2, 3] = m[1] * v
        c[:, 2, 4] = -m[0] * u
        c[:, 3, 1] = m[2] * w
        c[:, 3, 2] = -m[1] * v
        c[:, 3, 4] = m[5] * r
        c[:, 3, 5] = -m[4] * q
        c[:, 4, 0] = -m[2] * w
        c[:, 4, 2] = m[0] * u
        c[:, 4, 3] = -m[5] * r
        c[:, 4, 5] = m[3] * p
        c[:, 5, 0] = m[1] * v
        c[:, 5, 1] = -m[0] * u
        c[:, 5, 3] = m[4] * q
        c[:, 5, 4] = -m[3] * p
        return c

    def calculate_density_and_viscosity_forces(
        self,
        root_linvels_b: torch.Tensor,
        root_angvels_b: torch.Tensor,
    ):
        nu = torch.cat([root_linvels_b, root_angvels_b], dim=-1)
        dt = torch.as_tensor(self.dt, device=self.device, dtype=nu.dtype)
        dt = dt if dt.numel() == 1 else dt.flatten()[0]

        nu_dot = (nu - self._prev_nu) / dt
        if self._has_reset:
            reset_mask = self._reset_mask
            nu_dot[reset_mask] = 0.0
            reset_mask[reset_mask] = False
            self._has_reset = False

        lin = -self.linear_damping * nu
        quad = -self.quadratic_damping * torch.abs(nu) * nu
        damping = lin + quad

        added_mass_term = -self.added_mass * nu_dot
        c_a = self._coriolis_added(nu)
        coriolis_term = -torch.bmm(c_a, nu.unsqueeze(-1)).squeeze(-1)

        hydro_dynamic = damping + added_mass_term + coriolis_term
        self._prev_nu = nu.clone()

        return (
            hydro_dynamic[:, :3],
            hydro_dynamic[:, 3:],
            damping,
            added_mass_term,
            coriolis_term,
            nu_dot,
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self._prev_nu.zero_()
            self._reset_mask.fill_(True)
        else:
            self._prev_nu[env_ids] = 0.0
            self._reset_mask[env_ids] = True
        self._has_reset = True


def coriolis_added_direct(nu: torch.Tensor, added_mass: torch.Tensor) -> torch.Tensor:
    m = added_mass
    u, v, w, p, q, r = [nu[:, i] for i in range(6)]
    tau = torch.zeros_like(nu)
    tau[:, 0] = -m[2] * w * q + m[1] * v * r
    tau[:, 1] = m[2] * w * p - m[0] * u * r
    tau[:, 2] = -m[1] * v * p + m[0] * u * q
    tau[:, 3] = -(m[2] - m[1]) * v * w - (m[5] - m[4]) * q * r
    tau[:, 4] = (m[2] - m[0]) * u * w + (m[5] - m[3]) * p * r
    tau[:, 5] = (m[0] - m[1]) * u * v + (m[3] - m[4]) * p * q
    return tau  # equals (-C_A(nu) nu)


def reference_hydrodynamic(nu, nu_dot, model):
    lin = -model.linear_damping * nu
    quad = -model.quadratic_damping * torch.abs(nu) * nu
    damping = lin + quad
    added = -model.added_mass * nu_dot
    cor = coriolis_added_direct(nu, model.added_mass)
    return damping + added + cor


# -------------------- cases --------------------
dt = 0.02
device = torch.device("cpu")

cases = [
    ("1) steady surge u=0.5", [0.5, 0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0]),
    ("2) steady yaw r=0.3", [0, 0, 0, 0, 0, 0.3], [0, 0, 0, 0, 0, 0.3]),
    ("3) surge+yaw u=1.0 r=0.4", [1.0, 0, 0, 0, 0, 0.4], [1.0, 0, 0, 0, 0, 0.4]),
    ("4) heave+pitch w=0.6 q=0.5", [0, 0, 0.6, 0, 0.5, 0], [0, 0, 0.6, 0, 0.5, 0]),
    ("5) accel surge 0->1.2 in 0.02s", [1.2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    ("6) reset then jump to u=0.8", [0.8, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0]),
]

num_envs = len(cases)
model = FossenForceModels(num_envs=num_envs, device=device, dt=dt)

nu_prev = torch.tensor([c[2] for c in cases], dtype=torch.float32)
model._prev_nu = nu_prev.clone()

# reset env for case 6
model.reset(env_ids=torch.tensor([5], dtype=torch.long))

nu_now = torch.tensor([c[1] for c in cases], dtype=torch.float32)
F_b, T_b, damping, added_mass_term, coriolis_term, nu_dot = (
    model.calculate_density_and_viscosity_forces(
        root_linvels_b=nu_now[:, :3],
        root_angvels_b=nu_now[:, 3:],
    )
)

nu_dot_ref = (nu_now - nu_prev) / dt
nu_dot_ref[5] = 0.0  # reset behavior

hydro_ref = reference_hydrodynamic(nu_now, nu_dot_ref, model)
coriolis_ref = coriolis_added_direct(nu_now, model.added_mass)

rows = []
for i, (name, *_rest) in enumerate(cases):
    rows.append(
        {
            "case": name,
            "nu": nu_now[i].tolist(),
            "nu_dot": nu_dot[i].tolist(),
            "damping": damping[i].tolist(),
            "added_mass_term": added_mass_term[i].tolist(),
            "coriolis(matrix)": coriolis_term[i].tolist(),
            "coriolis(closed)": coriolis_ref[i].tolist(),
            "hydro(code)": torch.cat([F_b[i], T_b[i]]).tolist(),
            "hydro(ref)": hydro_ref[i].tolist(),
            "max_abs_err": float(
                (torch.cat([F_b[i], T_b[i]]) - hydro_ref[i]).abs().max()
            ),
        }
    )

df = pd.DataFrame(rows)
print(df[["case", "hydro(code)", "hydro(ref)", "max_abs_err"]])
