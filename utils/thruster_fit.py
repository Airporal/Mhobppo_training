'Fit the PWM-to-RPM mapping.'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


DATE_FILE = os.path.join(os.path.dirname(__file__), "../data/thruster/t200.csv")


csv = pd.read_csv(DATE_FILE)
pwm = csv["pwm 16V"].values
rpm = csv["rpm 16V"].values


rpm_fitted = np.zeros_like(rpm)


upper_mask = pwm > 1520
lower_mask = pwm < 1480



if np.sum(upper_mask) > 2:
    coeff_upper = np.polyfit(pwm[upper_mask], rpm[upper_mask], 2)
    rpm_fitted[upper_mask] = np.polyval(coeff_upper, pwm[upper_mask])
else:
    coeff_upper = [0, 0, 0, 0]


if np.sum(lower_mask) > 2:
    coeff_lower = np.polyfit(pwm[lower_mask], -rpm[lower_mask], 2)
    rpm_fitted[lower_mask] = -np.polyval(coeff_lower, pwm[lower_mask])
else:
    coeff_lower = [0, 0, 0, 0]





print("Upper segment coefficients (PWM>1520):", coeff_upper)
print("Lower segment coefficients (PWM<1480):", coeff_lower)


plt.figure(figsize=(8, 5))
plt.scatter(pwm, rpm, label="Original Data", color="blue")
plt.scatter(pwm, rpm_fitted, label="Fitted Data", color="red")
plt.axvline(1520, color="green", linestyle="--", label="Upper threshold")
plt.axvline(1480, color="orange", linestyle="--", label="Lower threshold")
plt.xlabel("PWM")
plt.ylabel("RPM")
plt.title("PWM -> RPM Mapping with Segmented Fit")
plt.legend()
plt.grid(True)
plt.show()
