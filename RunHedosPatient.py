from simulation import Patient
import os
import sys

# Add the simulation folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation")))


# === Step 1: Set your patient data folder ===
read_dir = r"input\patient"  # <-- change this to your actual path

# === Step 2: Define organs to include ===
organ_names = ["PTV"]  # <-- adjust based on your RTStruct

# === Step 3: Load patient data ===
patient = Patient()

patient.read_from_numpy(read_dir, organ_names, plot=True)

# === Step 4: Print mean dose for each organ ===
print("\n--- Mean Organ Doses ---")
for organ in organ_names:
    mean_dose = patient.get_mean_organ_dose(organ)
    print(f"{organ}: {mean_dose:.2f} Gy")

# === Step 5: Export DVHs to CSV ===
save_dir = os.path.join(read_dir, "DVHs")
os.makedirs(save_dir, exist_ok=True)
patient.write_dvh(save_dir, organ_names)

print(f"\nDVHs saved to: {save_dir}")
