import torch
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


def residuals_weighted(params, L_Real, L_Proj, weights, sdd, svd, vdd, manual_translation_x, manual_translation_y, manual_translation_z):

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

    shifted = L_Real - np.array([0, vdd, -manual_translation_z])
    rotated = shifted @ rot.T
    transformed = rotated + np.array([0, vdd, -manual_translation_z]) + np.array([tx, -ty, -tz])

    factor = sdd / (sdd - transformed[:, 1])
    L_Proj_pred = factor[:, None] * transformed[:, [0, 2]]

    # Raw residuals
    res = np.hstack([
        (L_Proj_pred[:, 0] - L_Proj[:, 0]),
        (L_Proj_pred[:, 1] - L_Proj[:, 1])
    ])

    # Apply sqrt(weights)
    w = np.repeat(np.sqrt(weights), 2)  # expand for x & y
    return w * res


def pose_estimation_weighted(L_Real, L_Proj, weights, sdd, svd, vdd, manual_translations_list):
    # Initial guess
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Remove NaNs
    valid_mask = np.isfinite(L_Proj).all(axis=1)
    L_Real_valid = L_Real[valid_mask]
    L_Proj_valid = L_Proj[valid_mask]
    weights_valid = weights[valid_mask]

    # Need ≥ 3 points
    if L_Real_valid.shape[0] < 3:
        return (
            np.array([np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan])
        )

    manual_translation_x = manual_translations_list[0, 0].item()
    manual_translation_y = manual_translations_list[0, 1].item()
    manual_translation_z = manual_translations_list[0, 2].item()

    result = least_squares(
        residuals_weighted,
        x0,
        args=(
            L_Real_valid, L_Proj_valid, weights_valid,
            sdd, svd, vdd,
            manual_translation_x, manual_translation_y, manual_translation_z
        ),
        method='lm',
        verbose=0
    )

    return result.x[:3], result.x[3:6]


def residuals_torch(params, L_Real, L_Proj, sdd, svd, vdd,
                    manual_translation_x, manual_translation_y, manual_translation_z):

    device = params.device

    rx, ry, rz, tx, ty, tz = params

    # Convert angles exactly like your numpy version
    rx_rad = torch.deg2rad(-ry)
    ry_rad = torch.deg2rad(rz)
    rz_rad = torch.deg2rad(rx)

    # -------------------------------------------------------
    # Rotation matrices using ONLY operations on tensors
    # -------------------------------------------------------
    Rx = torch.stack([
        torch.stack([torch.ones((), device=device), torch.zeros((), device=device), torch.zeros((), device=device)]),
        torch.stack([torch.zeros((), device=device), torch.cos(rx_rad), -torch.sin(rx_rad)]),
        torch.stack([torch.zeros((), device=device), torch.sin(rx_rad),  torch.cos(rx_rad)])
    ])

    Ry = torch.stack([
        torch.stack([ torch.cos(ry_rad), torch.zeros((), device=device), torch.sin(ry_rad)]),
        torch.stack([ torch.zeros((), device=device), torch.ones((), device=device), torch.zeros((), device=device)]),
        torch.stack([-torch.sin(ry_rad), torch.zeros((), device=device), torch.cos(ry_rad)])
    ])

    Rz = torch.stack([
        torch.stack([torch.cos(rz_rad), -torch.sin(rz_rad), torch.zeros((), device=device)]),
        torch.stack([torch.sin(rz_rad),  torch.cos(rz_rad), torch.zeros((), device=device)]),
        torch.stack([torch.zeros((), device=device), torch.zeros((), device=device), torch.ones((), device=device)])
    ])

    rot = Ry @ Rx @ Rz

    # -------------------------------------------------------
    # shift vector (use stack, NOT tensor([...]))
    # -------------------------------------------------------
    shift_vec = torch.stack([
        torch.zeros((), device=device),
        torch.tensor(vdd, dtype=torch.float32, device=device),
        -manual_translation_z
    ])

    shifted = L_Real - shift_vec
    rotated = shifted @ rot.T

    # -------------------------------------------------------
    # translation vector (use stack)
    # -------------------------------------------------------
    trans_vec = torch.stack([tx, -ty, -tz])
    transformed = rotated + shift_vec + trans_vec

    # -------------------------------------------------------
    # Projection
    # -------------------------------------------------------
    denom = (sdd - transformed[:, 1])
    denom = torch.clamp(denom, min=1e-6)
    factor = sdd / denom

    L_Proj_pred = factor.unsqueeze(-1) * transformed[:, [0, 2]]

    # residuals: flatten to match SciPy structure
    res = torch.cat([
        L_Proj_pred[:, 0] - L_Proj[:, 0],
        L_Proj_pred[:, 1] - L_Proj[:, 1]
    ])

    return res


def pose_estimation_torch(L_Real, L_Proj, sdd, svd, vdd, manual_translations_list,
                          iters=200, lr=5e-1):
    """
    Fully differentiable replacement for SciPy pose_estimation().
    
    L_Real: tensor (N,3)
    L_Proj: tensor (N,2)
    returns (rot, trans) each tensor of shape (3,)
    """

    device = L_Real.device

    # mask out invalid 2D coords (same as before)
    valid_mask = torch.isfinite(L_Proj).all(dim=1)
    if valid_mask.sum() < 3:
        return (torch.full((3,), float("nan"), device=device),
                torch.full((3,), float("nan"), device=device))

    L_Real_valid = L_Real[valid_mask]
    L_Proj_valid = L_Proj[valid_mask]

    # initial guess (same as numpy version)
    params = torch.zeros(6, dtype=torch.float32, device=device, requires_grad=True)

    manual_translation_x = manual_translations_list[0, 0]
    manual_translation_y = manual_translations_list[0, 1]
    manual_translation_z = manual_translations_list[0, 2]

    optimizer = torch.optim.Adam([params], lr=lr)

    for _ in range(iters):
        optimizer.zero_grad()

        res = residuals_torch(
            params,
            L_Real_valid,
            L_Proj_valid,
            sdd, svd, vdd,
            manual_translation_x,
            manual_translation_y,
            manual_translation_z
        )

        loss = (res ** 2).mean()
        loss.backward()
        optimizer.step()

    rot = params[:3].clone().detach()
    trans = params[3:].clone().detach()
    return rot, trans


def residuals_weighted_torch(params, L_Real, L_Proj, weights,
                             sdd, svd, vdd,
                             manual_translation_x, manual_translation_y, manual_translation_z):

    device = params.device

    rx, ry, rz, tx, ty, tz = params

    # Convert angles to radians (same logic as numpy version)
    rx_rad = torch.deg2rad(-ry)
    ry_rad = torch.deg2rad(rz)
    rz_rad = torch.deg2rad(rx)

    # -------------------------------------------------------
    # Rotation matrices
    # -------------------------------------------------------
    Rx = torch.stack([
        torch.stack([torch.ones((), device=device), torch.zeros((), device=device), torch.zeros((), device=device)]),
        torch.stack([torch.zeros((), device=device), torch.cos(rx_rad), -torch.sin(rx_rad)]),
        torch.stack([torch.zeros((), device=device), torch.sin(rx_rad),  torch.cos(rx_rad)])
    ])

    Ry = torch.stack([
        torch.stack([ torch.cos(ry_rad), torch.zeros((), device=device), torch.sin(ry_rad)]),
        torch.stack([ torch.zeros((), device=device), torch.ones((), device=device), torch.zeros((), device=device)]),
        torch.stack([-torch.sin(ry_rad), torch.zeros((), device=device), torch.cos(ry_rad)])
    ])

    Rz = torch.stack([
        torch.stack([torch.cos(rz_rad), -torch.sin(rz_rad), torch.zeros((), device=device)]),
        torch.stack([torch.sin(rz_rad),  torch.cos(rz_rad), torch.zeros((), device=device)]),
        torch.stack([torch.zeros((), device=device), torch.zeros((), device=device), torch.ones((), device=device)])
    ])

    rot = Ry @ Rx @ Rz

    # -------------------------------------------------------
    # Shift vector (exact SciPy equivalent)
    # -------------------------------------------------------
    shift_vec = torch.stack([
        torch.zeros((), device=device),
        torch.tensor(vdd, dtype=torch.float32, device=device),
        -manual_translation_z
    ])

    shifted = L_Real - shift_vec
    rotated = shifted @ rot.T

    # translation
    trans_vec = torch.stack([tx, -ty, -tz])
    transformed = rotated + shift_vec + trans_vec

    # -------------------------------------------------------
    # Projection
    # -------------------------------------------------------
    denom = torch.clamp((sdd - transformed[:, 1]), min=1e-6)
    factor = sdd / denom
    L_Proj_pred = factor.unsqueeze(-1) * transformed[:, [0, 2]]

    # -------------------------------------------------------
    # Raw residuals (flatten same as SciPy)
    # -------------------------------------------------------
    res = torch.cat([
        L_Proj_pred[:, 0] - L_Proj[:, 0],
        L_Proj_pred[:, 1] - L_Proj[:, 1]
    ])

    # -------------------------------------------------------
    # Apply sqrt(weights) EXACTLY like SciPy:
    # w = repeat(sqrt(weights), 2)
    # -------------------------------------------------------
    w = torch.sqrt(weights)
    w = torch.cat([w, w])   # repeat for x and y residuals

    weighted_res = w * res
    return weighted_res


def pose_estimation_weighted_torch(L_Real, L_Proj, weights,
                                   sdd, svd, vdd, manual_translations_list,
                                   iters=200, lr=5e-1):
    """
    Standard PyTorch solver (Not end-to-end differentiable).
    Used for inference or debugging.
    """
    device = L_Real.device

    # 1. Masking
    valid_mask = torch.isfinite(L_Proj).all(dim=1)
    if valid_mask.sum() < 3:
        return (torch.full((3,), float("nan"), device=device),
                torch.full((3,), float("nan"), device=device))

    L_Real_valid = L_Real[valid_mask]
    L_Proj_valid = L_Proj[valid_mask]
    
    # --- CRITICAL FIX: DETACH WEIGHTS ---
    # We detach so the optimization loop doesn't try to backprop to the U-Net
    weights_valid = weights[valid_mask].detach() 

    # 2. Setup
    params = torch.zeros(6, dtype=torch.float32, device=device, requires_grad=True)
    
    manual_tx = manual_translations_list[0, 0]
    manual_ty = manual_translations_list[0, 1]
    manual_tz = manual_translations_list[0, 2]

    optimizer = torch.optim.Adam([params], lr=lr)

    # 3. Optimization
    for _ in range(iters):
        optimizer.zero_grad()

        res = residuals_weighted_torch(
            params,
            L_Real_valid,
            L_Proj_valid,
            weights_valid, # Now detached
            sdd, svd, vdd,
            manual_tx, manual_ty, manual_tz
        )

        loss = (res ** 2).mean()
        loss.backward()
        optimizer.step()

    rot = params[:3].clone().detach()
    trans = params[3:].clone().detach()
    return rot, trans


def pose_estimation_differentiable(L_Real, L_Proj, sdd, svd, vdd, manual_translations_list,
                                   iters=200, lr=0.1, momentum=0.9):
    """
    High-Precision Differentiable Solver with Momentum and LR Decay.
    """
    device = L_Real.device
    
    # 1. Masking
    valid_mask = torch.isfinite(L_Proj).all(dim=1)
    if valid_mask.sum() < 3: 
        nan_vec = torch.full((3,), float('nan'), device=device, requires_grad=True)
        return nan_vec, nan_vec

    L_Real_valid = L_Real[valid_mask]
    L_Proj_valid = L_Proj[valid_mask]
    
    # 2. Initialization
    params = torch.zeros(6, dtype=torch.float32, device=device)
    
    # Initialize Momentum Buffer (Velocity)
    velocity = torch.zeros_like(params)
    
    manual_tx = manual_translations_list[0, 0]
    manual_ty = manual_translations_list[0, 1]
    manual_tz = manual_translations_list[0, 2]

    # 3. Optimization Loop
    for i in range(iters):
        iter_params = params.detach().requires_grad_(True)
        
        res = residuals_torch(
            iter_params,
            L_Real_valid,
            L_Proj_valid,
            sdd, svd, vdd,
            manual_tx, manual_ty, manual_tz
        )
        
        loss = (res ** 2).mean()

        grads = torch.autograd.grad(loss, iter_params, create_graph=True)[0]
        
        if torch.isnan(grads).any():
            break

        # Clip Gradients (Stability)
        grads = torch.clamp(grads, min=-100.0, max=100.0)

        # -----------------------------------------------------------
        # UPGRADE: Momentum + LR Decay
        # -----------------------------------------------------------
        
        # 1. Learning Rate Decay (Cosine-ish / Linear decay)
        # Drops LR from 'lr' down to 'lr/10' by the end of the loop
        # This allows fast movement early, and fine precision later.
        current_lr = lr * (1.0 - 0.9 * (i / iters))
        
        # 2. Update Velocity (Momentum)
        # v = m * v + g
        # This pushes the solver through flat areas (like Translation Y)
        velocity = momentum * velocity + grads
        
        # 3. Update Params
        params = params - current_lr * velocity

    rot = params[:3]
    trans = params[3:]
    return rot, trans


def pose_estimation_weighted_differentiable(L_Real, L_Proj, weights, sdd, svd, vdd, 
                                            manual_translations_list,
                                            iters=300, landmark_type='ground truth'): # Increased iters slightly
    """
    Two-Stage Differentiable Solver (Coarse-to-Fine).
    Phase 1: High LR for rapid convergence (0-200 iters).
    Phase 2: Low LR for precision (200-300 iters).
    """
    device = L_Real.device
    
    # 1. Masking
    valid_mask = torch.isfinite(L_Proj).all(dim=1)
    if valid_mask.sum() < 3: 
        nan_vec = torch.full((3,), float('nan'), device=device, requires_grad=True)
        return nan_vec, nan_vec

    L_Real_valid = L_Real[valid_mask]
    L_Proj_valid = L_Proj[valid_mask]
    weights_valid = weights[valid_mask]
    
    # 2. Initialize Params
    params = torch.zeros(6, dtype=torch.float32, device=device)
    
    manual_tx = manual_translations_list[0, 0]
    manual_ty = manual_translations_list[0, 1]
    manual_tz = manual_translations_list[0, 2]

    # 3. Adam State
    exp_avg = torch.zeros_like(params)
    exp_avg_sq = torch.zeros_like(params)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    # 4. Optimization Loop
    for i in range(iters):
        # --- SCHEDULER ---
        # If we are in the first 200 steps, use high LR (0.5) to move fast.
        # If we are in the last 100 steps, use low LR (0.05) to finetune.
        if i < 200:
            current_lr = 0.5 
        else:
            current_lr = 0.05

        iter_params = params.detach().requires_grad_(True)
        
        res = residuals_weighted_torch(
            iter_params,
            L_Real_valid,
            L_Proj_valid,
            weights_valid,
            sdd, svd, vdd,
            manual_tx, manual_ty, manual_tz
        )
        
        loss = (res ** 2).mean()

        grads = torch.autograd.grad(loss, iter_params, create_graph=True)[0]
        
        if torch.isnan(grads).any():
            break

        # Relaxed Clipping (Allow big moves)
        grads = torch.clamp(grads, min=-100.0, max=100.0)

        # --- MANUAL ADAM ---
        exp_avg = beta1 * exp_avg + (1 - beta1) * grads
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grads ** 2)
        
        step_idx = i + 1
        bias_correction1 = 1 - beta1 ** step_idx
        bias_correction2 = 1 - beta2 ** step_idx
        
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)) + eps
        
        step_size = (exp_avg / bias_correction1) / denom
        
        # Update
        params = params - current_lr * step_size

    rot = params[:3]
    trans = params[3:]
    return rot, trans