import numpy as np

from scipy.optimize import least_squares


# -ry, rz, rx, tx, -ty, -tz
def residuals(params, L_Real, L_Proj, sdd, svd, vdd):
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

    shifted = L_Real - np.array([0, vdd, 0])
    rotated = shifted @ rot.T
    transformed = rotated + np.array([0, vdd, 0]) + np.array([tx, -ty, -tz])

    factor = sdd / (sdd - transformed[:, 1])
    L_Proj_pred = factor[:, None] * transformed[:, [0, 2]]

    res = np.hstack([(L_Proj_pred[:, 0] - L_Proj[:, 0]), (L_Proj_pred[:,1] - L_Proj[:, 1])])

    return res


def pose_estimation(L_Real, L_Proj, sdd, svd, vdd):
    # Initial guess
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Mask Out Nans in L_Proj
    valid_mask = np.isfinite(L_Proj).all(axis=1)

    L_Real_valid = L_Real[valid_mask]
    L_Proj_valid = L_Proj[valid_mask]

    # Call least_squares
    result = least_squares(
        residuals, x0, args=(L_Real_valid, L_Proj_valid, sdd, svd, vdd),
        method='lm',
        verbose=0,
    )

    return result.x[:3], result.x[3:6]