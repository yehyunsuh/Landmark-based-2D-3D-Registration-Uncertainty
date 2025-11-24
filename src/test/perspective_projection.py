import numpy as np


def apply_transformation(points, rx, ry, rz, tx, ty, tz, center):
    # Convert angles from degrees to radians
    rx_rad = np.deg2rad(rx)
    ry_rad = np.deg2rad(ry)
    rz_rad = np.deg2rad(rz)
    
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

    shifted = points - center
    rotated = shifted @ rot.T
    transformed = rotated + center + np.array([tx, ty, tz])

    return transformed


def project_point(S_T, H=1000):
    factor = H / (H - S_T[:, 1])
    return factor[:, None] * S_T[:, [0, 2]]