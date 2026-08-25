'Test the thruster simulation.'
try:
    from thrusterModels.thruster import ThrusterCfg, Thruster
except ImportError or ModuleNotFoundError:
    
    import isaacsim 
    import os

    UTILS_PATH = os.path.dirname(os.path.abspath(__file__)) + "/.."
    import sys
    sys.path.append(UTILS_PATH)
    from thrusterModels.thruster import ThrusterCfg, Thruster
from isaaclab.utils import configclass
import torch


@configclass
class TestThrusterCfg(ThrusterCfg):
    dynamic_type = "FirstOrder"
    tau = 0.05
    conversion_type = "Basic"
    rotor_constant = 0.1 / 100.0
    thruster_configure = "BLUEROV_THRUSTER_CONFIG"


# class TestThruster(Thruster):
#     cfg: TestThrusterCfg

#     def __init__(self, numEnvs: int, cfg: ThrusterCfg, device: torch.device):
#         super().__init__(numEnvs=numEnvs, cfg=cfg, device=device)


def main():
    test_thruster_cfg = TestThrusterCfg()
    num_envs = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thruster = Thruster(num_envs, test_thruster_cfg, device)
    print(thruster.cfg.thruster_configure)
    
    thruster.reset_all()
    print("Initial thruster state(3x8):", thruster.getVelocity())
    dt = torch.tensor(1 / 100, device=device)
    cmd1 = torch.tensor(
        [
            [0.25, -0.25, 0.25, -0.25, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.25, 0.25, -0.25, 0.25],
            [-0.25, 0.25, -0.25, 0.25, -0.25, -0.25, 0.25, -0.25],
        ],
        dtype=torch.float32,
        device=device,
    )
    thrustForce, wrench_b = thruster.update(cmd1, dt)
    print("Thruster wrench in base_link(FRD 3x6):", wrench_b)


if __name__ == "__main__":
    main()
