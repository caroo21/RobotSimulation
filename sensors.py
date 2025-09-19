# sensors.py
from dataclasses import dataclass
import numpy as np
import pybullet as p

# get rgbd frame: compute projection, get cameraimage, depth in meter
# intrinsics + inverted View Matrix -> Kamera coordinates

@dataclass
class CameraConfig:
    eye: tuple = (0.7, 0.0, 1.1)
    target: tuple = (0.0, 0.0, 0.6)
    up: tuple = (0.0, 0.0, 1.0)
    width: int = 320
    height: int = 240
    fov_deg: float = 60.0
    near: float = 0.01
    far: float = 3.0
    renderer: int = None  # If None, will be chosen based on connection method
    max_range: float = None  # cap useful depth in meters, e.g., 2.0

def get_rgbd_frame(
    eye=(0.7, 0.0, 1.1),
    target=(0.0, 0.0, 0.6),
    up=(0.0, 0.0, 1.0),
    width=320,
    height=240,
    fov_deg=60.0,
    near=0.01,
    far=3.0,
    renderer=None,
    return_matrices=False
):

    if not p.isConnected():
        raise RuntimeError("get_rgbd_frame called without PyBullet connection.")

    aspect = float(width) / float(height)
    view_mat = p.computeViewMatrix(eye, target, up)
    proj_mat = p.computeProjectionMatrixFOV(fov=fov_deg, aspect=aspect, nearVal=near, farVal=far)

    # Choose renderer if not specified
    if renderer is None:
        method = p.getConnectionInfo().get("connectionMethod")
        renderer = p.ER_BULLET_HARDWARE_OPENGL if method == p.GUI else p.ER_TINY_RENDERER

    w, h, rgba, depth_buffer, seg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view_mat, projectionMatrix=proj_mat,
        renderer=renderer
    )

    rgba_img = np.reshape(np.array(rgba, dtype=np.uint8), (h, w, 4))
    rgb = rgba_img[:, :, :3].copy()

    depth_buf = np.array(depth_buffer, dtype=np.float32).reshape(h, w)
    # Depth buffer (0..1) → meters along view ray
    depth_m = (2.0 * near * far) / (far + near - (2.0 * depth_buf - 1.0) * (far - near))

    seg = np.array(seg).reshape(h, w)

    if return_matrices: 
        return rgb, depth_m, seg, view_mat, proj_mat, (near, far, width, height, fov_deg)
    return rgb, depth_m, seg

def get_rgbd_with_config(cfg: CameraConfig, return_matrices=False):
    renderer = cfg.renderer
    if renderer is None:
        method = p.getConnectionInfo().get("connectionMethod")
        renderer = p.ER_BULLET_HARDWARE_OPENGL if method == p.GUI else p.ER_TINY_RENDERER

    return get_rgbd_frame(
        eye=cfg.eye,
        target=cfg.target,
        up=cfg.up,
        width=cfg.width,
        height=cfg.height,
        fov_deg=cfg.fov_deg,
        near=cfg.near,
        far=cfg.far,
        renderer=renderer,
        return_matrices=return_matrices
    )

def show_rgb_depth():
    import matplotlib.pyplot as plt
    rgb, depth_m, _ = get_rgbd_frame()
    plt.figure(); plt.imshow(rgb); plt.title("RGB"); plt.axis("off")
    plt.figure(); plt.imshow(depth_m, cmap="inferno"); plt.title("Depth (m)"); plt.colorbar()
    plt.show()

def camera_intrinsics_from_fov(width, height, fov_deg):
    fov_rad =np.deg2rad(fov_deg)
    fy = (height/2.0)/np.tan(fov_rad/2.0)
    fx = fy
    cx = (width-1)/2.0
    cy = (height-1)/2.0
    return fx, fy, cx, cy

def invert_view_matrix(view_mat):
    V=np.array(view_mat, dtype=np.float32).reshape(4,4).T
    R_inv=V[:3, :3].T
    t=V[:3, 3]
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv  # camera-to-world

def depth_to_pointcloud_world(depth_m, view_mat, width, height, fov_deg, max_range=None):
    fx, fy, cx, cy = camera_intrinsics_from_fov(width, height, fov_deg)
    h, w = depth_m.shape
    assert h == height and w == width

    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    Z = depth_m
    valid = np.isfinite(Z) & (Z > 0)

    if max_range is not None:
        valid &= (Z <= max_range)

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    pts_cam = np.stack([X, Y, Z, np.ones_like(Z, dtype=np.float32)], axis=-1)  # (H, W, 4)
    T_c2w = invert_view_matrix(view_mat)

    pts_cam_flat = pts_cam[valid].reshape(-1, 4).T  # (4, N)
    pts_world = (T_c2w @ pts_cam_flat).T[:, :3]     # (N, 3)
    return pts_world, valid

def rgbd_to_colored_pointcloud(cfg: CameraConfig, max_points: int = 50000):

    # 1) Capture with matrices
    rgb, depth_m, seg, view_mat, proj_mat, (near, far, W, H, fov) = get_rgbd_with_config(cfg, return_matrices=True)

    # 2) Back-project to world
    pts_world, valid_mask = depth_to_pointcloud_world(depth_m, view_mat, W, H, fov, cfg.max_range)

    # 3) Colors for valid points
    rgb_flat = rgb.reshape(-1, 3)
    valid_flat = valid_mask.reshape(-1)
    colors = rgb_flat[valid_flat].astype(np.float32) / 255.0

    # 4) Optional downsampling for speed
    N = pts_world.shape[0]
    if max_points is not None and N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        pts_world = pts_world[idx]
        colors = colors[idx]

    meta = {
        "view_mat": view_mat,
        "proj_mat": proj_mat,
        "width": W,
        "height": H,
        "fov_deg": fov,
        "near": near,
        "far": far,
    }
    return pts_world, colors, meta

def filter_table_by_height(pts_world, z_table=0.52, band=0.02):
    z = pts_world[:, 2]
    keep_mask = z > (z_table + band)
    return pts_world[keep_mask], keep_mask

def filter_table_by_height_colored(pts_world, colors, z_table=0.52, band=0.02):
    pts_keep, mask = filter_table_by_height(pts_world, z_table, band)
    cols_keep = colors[mask]
    return pts_keep, cols_keep, mask
