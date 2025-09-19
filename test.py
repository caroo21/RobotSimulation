import numpy as np
from sensors import CameraConfig, get_rgbd_frame, get_rgbd_with_config, depth_to_pointcloud_world, rgbd_to_colored_pointcloud, filter_table_by_height_colored

def filter_table_perfect(pts_world, colors):
    # Entfernt exakt deine Tisch-Box: -0.5≤x≤0.5, -0.5≤y≤0.5, 0.48≤z≤0.52
    table_pos = [0, 0, 0.5]
    half_extents = [0.5, 0.5, 0.02]
    margin = 0.01
    
    x, y, z = pts_world[:, 0], pts_world[:, 1], pts_world[:, 2]
    
    inside_table = ((x >= -0.51) & (x <= 0.51) & 
                   (y >= -0.51) & (y <= 0.51) & 
                   (z >= 0.47) & (z <= 0.53))
    
    keep_mask = ~inside_table
    return pts_world[keep_mask], colors[keep_mask], keep_mask


def test_filter_table_once():
    cfg = CameraConfig(max_range=2.0)
    pts, cols, meta = rgbd_to_colored_pointcloud(cfg, max_points=80000)
    print("Raw cloud:", pts.shape[0])

    z_table = 0
    # ÄNDERE DIESE ZEILE:
    pts_obj, cols_obj, mask = filter_table_by_height_colored(pts, cols, z_table=z_table, band=0.005)  # 0.005 statt 0.02!
    print("Above table:", pts_obj.shape[0])

    print(f"Z-Bereich: {pts[:,2].min():.3f} - {pts[:,2].max():.3f}")
    print(f"Punkte vor Filter: {len(pts)}")
    print(f"Punkte nach Filter: {len(pts_obj)}")
    print(f"Entfernt: {len(pts) - len(pts_obj)}")


    # Quick plot before/after
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pts[:,0], pts[:,1], pts[:,2], c=cols, s=0.5)
        ax1.set_title("Raw cloud")
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(pts_obj[:,0], pts_obj[:,1], pts_obj[:,2], c=cols_obj, s=0.8)
        ax2.set_title("Filtered (above table)")
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

        # Optional: consistent view
        ax1.view_init(elev=20, azim=-60)
        ax2.view_init(elev=20, azim=-60)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)

def test_colored_pointcloud_once():
    cfg = CameraConfig(max_range=1.5)
    pts, cols, meta = rgbd_to_colored_pointcloud(cfg, max_points=30000)
    print(f"Colored point cloud: {pts.shape[0]} points")

    # Quick 3D plot
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        s = 1 if pts.shape[0] <= 30000 else 0.5 
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=cols, s=s)
        ax.set_title("Colored Point Cloud (world)")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("3D color plot skipped:", e)


def test_pointcloud_once():
    cfg = CameraConfig()
    rgb, depth_m, seg, view_mat, proj_mat, (near, far, W, H, fov) = get_rgbd_with_config(cfg, return_matrices=True)
    pts_world, valid_mask = depth_to_pointcloud_world(depth_m, view_mat, W, H, fov)
    print("Point cloud:", pts_world.shape[0], "points")
    import numpy as np
    pts_world_new = np.flip(pts_world, axis=1)  # Flip für bessere Ansicht)
    #pts_world_new = np.flip(pts_world_new)
    # Optional quick scatter (downsample for speed)
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        n = pts_world_new.shape[0]
        if n > 20000:
            idx = np.random.choice(n, 20000, replace=False)
            pts = pts_world_new[idx]
        else:
            pts = pts_world_new
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
        ax.set_title("Point Cloud (world)")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("3D plot skipped:", e)


def show_rgb_depth():
    import matplotlib.pyplot as plt
    rgb, depth_m, seg = get_rgbd_frame()  # uses defaults from camera_utils
    depth_vis = depth_m.copy()
    depth_vis[np.isinf(depth_vis) | (depth_vis <= 0)] = np.nan

    import matplotlib
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(rgb); axs[0].set_title("RGB"); axs[0].axis("off")
    im1 = axs[1].imshow(depth_vis, cmap="inferno"); axs[1].set_title("Depth (m)"); axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    axs[2].imshow(seg); axs[2].set_title("Segmentation IDs"); axs[2].axis("off")
    plt.tight_layout(); plt.show()
