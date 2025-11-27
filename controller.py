import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    delta_cur = state[2]
    v_cur = state[3]
    
    delta_ref = desired[0]
    v_ref = desired[1]
    
    kp_steer = 8.0  
    kp_acc = 25.0
    
    v_delta = kp_steer * (delta_ref - delta_cur)
    a = kp_acc * (v_ref - v_cur)
    
    return np.array([v_delta, a])

def get_target_point(path, position, lookahead_dist, closest_idx):
    n_points = len(path)
    target_idx = closest_idx
    cum_dist = 0.0
    
    for i in range(n_points):
        curr = (closest_idx + i) % n_points
        next_p = (closest_idx + i + 1) % n_points
        
        segment_len = np.linalg.norm(path[next_p] - path[curr])
        cum_dist += segment_len
        
        if cum_dist >= lookahead_dist:
            target_idx = next_p
            break
            
    return path[target_idx]

def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    position = state[0:2]
    v = state[3]
    phi = state[4]
    L = parameters[0] 
    
    path = racetrack.centerline
    
    diffs = path - position
    dists_sq = np.sum(diffs**2, axis=1)
    closest_idx = np.argmin(dists_sq)
    min_dist = np.sqrt(dists_sq[closest_idx])
    
    steer_base = 6.0 
    steer_factor = 0.35 * v
    steer_recovery = 1.0 * min_dist 
    dist_steer = steer_base + steer_factor + steer_recovery
    
    brake_factor = 2.0 * v
    dist_brake = 10.0 + brake_factor
    
    target_steer = get_target_point(path, position, dist_steer, closest_idx)
    target_brake = get_target_point(path, position, dist_brake, closest_idx)
    
    dx = target_steer[0] - position[0]
    dy = target_steer[1] - position[1]
    phi_desired = np.arctan2(dy, dx)
    phi_err = wrap_to_pi(phi_desired - phi)
    
    # High-Speed Steering Law
    delta_ref = 1.8 * (L / dist_steer) * phi_err
    delta_ref = np.clip(delta_ref, -0.6, 0.6)
    
    dx_brake = target_brake[0] - position[0]
    dy_brake = target_brake[1] - position[1]
    phi_brake = np.arctan2(dy_brake, dx_brake)
    phi_err_brake = wrap_to_pi(phi_brake - phi)
    
    max_speed = 84.0
    a_lat_max = 4.5
    
    if abs(delta_ref) > 0.02: 
        R_turn = L / abs(delta_ref)
        v_limit_geo = np.sqrt(a_lat_max * R_turn)
    else:
        v_limit_geo = max_speed
    
    # Avoid division by zero with small epsilon
    safe_error = max(abs(phi_err_brake), 0.001)
    
    estimated_radius = dist_brake / (2.0 * np.sin(safe_error))
    
    v_limit_align = np.sqrt(a_lat_max * abs(estimated_radius))
    
    v_target = min(v_limit_geo, v_limit_align, max_speed)
    
    # Minimum Speed Floor
    v_ref = max(v_target, 20.5)
    
    return np.array([delta_ref, v_ref])
