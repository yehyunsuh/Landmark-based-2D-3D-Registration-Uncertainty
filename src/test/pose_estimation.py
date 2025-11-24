import numpy as np

from scipy.optimize import least_squares


def residuals(params, L_Real, L_Proj, sdd, svd, vdd, manual_translation_x, manual_translation_y, manual_translation_z):
    rx, ry, rz, tx, ty, tz = params

    # Convert angles from degrees to radians
    rx_rad = np.deg2rad(-ry)
    ry_rad = np.deg2rad(rz)
    rz_rad = np.deg2rad(rx)
    
    # Rotation matrices for X, Y, Z
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_rad), -np.sin(rx_rad)],
        [0, np.sin(rx_rad),  np.cos(rx_rad)]
    ])
    Ry = np.array([
        [ np.cos(ry_rad), 0, np.sin(ry_rad)],
        [ 0, 1, 0],
        [-np.sin(ry_rad), 0, np.cos(ry_rad)]
    ])
    Rz = np.array([
        [np.cos(rz_rad), -np.sin(rz_rad), 0],
        [np.sin(rz_rad),  np.cos(rz_rad), 0],
        [0, 0, 1]
    ])
    rot = Ry @ Rx @ Rz

    # print("\nL_REAL: ", L_Real[:5])
    shifted = L_Real - np.array([0, vdd, -manual_translation_z])
    rotated = shifted @ rot.T
    transformed = rotated + np.array([0, vdd, -manual_translation_z]) + np.array([tx, -ty, -tz])

    factor = sdd / (sdd - transformed[:, 1])
    L_Proj_pred = factor[:, None] * transformed[:, [0, 2]]

    # print(f"\nL_Proj_pred: {L_Proj_pred[:5]}")
    # print(f"\nL_Proj: {L_Proj[:5]}")
    # exit()

    res = np.hstack([(L_Proj_pred[:, 0] - L_Proj[:, 0]), (L_Proj_pred[:,1] - L_Proj[:, 1])])
    return res


def pose_estimation(L_Real, L_Proj, sdd, svd, vdd, manual_translations_list):
    # === Initial guess ===
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Mask Out Nans in L_Proj
    valid_mask = np.isfinite(L_Proj).all(axis=1)

    L_Real_valid = L_Real[valid_mask]
    L_Proj_valid = L_Proj[valid_mask]

    # --- NEW: guard for too few points ---
    if L_Real_valid.shape[0] < 3:
        # Not enough constraints for 6-DOF pose using LM
        # Return NaNs so caller can skip this case
        return (
            np.array([np.nan, np.nan, np.nan], dtype=float),
            np.array([np.nan, np.nan, np.nan], dtype=float),
        )

    manual_translation_x = manual_translations_list[0, 0].item()
    manual_translation_y = manual_translations_list[0, 1].item()
    manual_translation_z = manual_translations_list[0, 2].item()

    result = least_squares(
        residuals, 
        x0, 
        args=(
            L_Real_valid, L_Proj_valid, 
            sdd, svd, vdd, 
            manual_translation_x, manual_translation_y, manual_translation_z
        ),
        method='lm',
        verbose=0,
    )

    return result.x[:3], result.x[3:6]