import torch
import yaml


num_centers = 16
input_dim = 6
device = torch.device("cuda")
dtype = torch.float32
center_range = [-8.0, 8.0, -10.0, 10.0]  # x,y,z,theta



cen = torch.empty(num_centers, input_dim, device=device, dtype=dtype)
cen[:, 0:3].uniform_(center_range[0], center_range[1])
cen[:, 3:].uniform_(center_range[2], center_range[3])
print(cen)
centers_yaml = "centers.yaml"
with open(centers_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump({"centers": cen.cpu().tolist()}, f)

with open(centers_yaml, "r", encoding="utf-8") as f:
    loaded_centers = yaml.safe_load(f)["centers"]

loaded_cen = torch.tensor(loaded_centers, device=device, dtype=dtype)
print(loaded_cen)
assert torch.equal(cen, loaded_cen), f"Failed to save and reload centers from {centers_yaml}"
