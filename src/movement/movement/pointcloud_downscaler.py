import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import cv2
import sensor_msgs_py.point_cloud2 as pc2


W_H_MIN, W_H_MAX = 0,   179
W_S_MIN, W_S_MAX = 0,    25
W_V_MIN, W_V_MAX = 110, 170

Y_H_MIN, Y_H_MAX = 15,  40
Y_S_MIN, Y_S_MAX = 80, 255
Y_V_MIN, Y_V_MAX = 120, 255

B_S_MAX = 70
B_V_MAX = 70

VOXEL_SIZE = 0.05

LOG = False


def _voxel_downsample(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """
    Grid-subsample (aka voxel filter) in O(N log N) with a cheap integer hash.

    Keeps the first point that lands in each voxel.
    """
    if xyz.size == 0:
        return xyz  # nothing to do

    ijk = np.floor_divide(xyz, voxel).astype(np.int32)          # (N,3)
    # Fast 3-D integer hash using large distinct primes
    keys = (
        ijk[:, 0] * 73856093 ^       # xor avoids overflow & mixes bits
        ijk[:, 1] * 19349663 ^
        ijk[:, 2] * 83492791
    ).astype(np.int64)

    _, first_idx = np.unique(keys, return_index=True)
    return np.ascontiguousarray(xyz[first_idx])


def fast_xyz_white_yellow(msg, log):
    """
    Slice the incoming PointCloud2 into four voxel‑filtered clouds:
      1) white lane‑paint points   (–2.0 < z < –1.3)   — HSV band W_*
      2) yellow lane‑paint points (–2.0 < z < –1.3)   — HSV band Y_*
      3) not‑black points (all finite z)               — useful for viz / debug
      4) mid‑slice points       (–1.2 < z < –0.7)      — for extra heuristics
    """
    n_pts  = msg.width * msg.height
    step   = msg.point_step
    endian = '>' if msg.is_bigendian else '<'
    if LOG:
        log.info(f"Incoming cloud: {n_pts} points")

    # 1. reshape raw buffer → (N, fields)
    cloud_f32 = np.frombuffer(msg.data, dtype=endian + 'f4')\
                   .reshape(n_pts, step // 4)
    fld = {f.name: f.offset // 4 for f in msg.fields}
    xs, ys, zs = (cloud_f32[:, fld[k]] for k in ('x', 'y', 'z'))
    rgbf       = cloud_f32[:, fld['rgb']]

    finite = np.isfinite(zs)
    if not finite.any():
        empty = np.empty((0, 3), np.float32)
        return empty, empty, empty, empty

    # 2. unpack BGR → HSV (vectorised, one conversion)
    xs, ys, zs, rgbf = xs[finite], ys[finite], zs[finite], rgbf[finite]
    rgb_u32 = rgbf.view(np.uint32)
    b = (rgb_u32 & 0xFF).astype(np.uint8)
    g = ((rgb_u32 >> 8)  & 0xFF).astype(np.uint8)
    r = ((rgb_u32 >> 16) & 0xFF).astype(np.uint8)

    hsv_all = cv2.cvtColor(
        np.stack([b, g, r], axis=1).reshape(-1, 1, 3),
        cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)
    h_all, s_all, v_all = hsv_all[:, 0], hsv_all[:, 1], hsv_all[:, 2]

    # 3. whole‑cloud "not black"
    not_black_m   = ~((s_all <= B_S_MAX) & (v_all <= B_V_MAX))
    not_black_xyz = np.column_stack((xs[not_black_m],
                                     ys[not_black_m],
                                     zs[not_black_m])).astype(np.float32)

    # 4. mid‑slice for extra debug / algorithms
    mid_m   = (zs > -1.2) & (zs < -0.7)
    mid_xyz = np.column_stack((xs[mid_m],
                               ys[mid_m],
                               zs[mid_m])).astype(np.float32)

    # 5. ground slice for white / yellow
    ground_m = (zs > -2.0) & (zs < -1.3)
    if not ground_m.any():
        # still return voxel‑filtered aux clouds
        return (np.empty((0, 3), np.float32),
                np.empty((0, 3), np.float32),
                _voxel_downsample(not_black_xyz, VOXEL_SIZE),
                _voxel_downsample(mid_xyz,      VOXEL_SIZE))

    xs_g, ys_g, zs_g = xs[ground_m], ys[ground_m], zs[ground_m]
    h_g,  s_g,  v_g  = h_all[ground_m], s_all[ground_m], v_all[ground_m]

    # 5a. HSV‑based white & yellow tests (your final ranges)
    is_white_g = (
        (h_g >= W_H_MIN) & (h_g <= W_H_MAX) &
        (s_g >= W_S_MIN) & (s_g <= W_S_MAX) &
        (v_g >= W_V_MIN) & (v_g <= W_V_MAX)
    )
    is_yellow_g = (
        (h_g >= Y_H_MIN) & (h_g <= Y_H_MAX) &
        (s_g >= Y_S_MIN) & (s_g <= Y_S_MAX) &
        (v_g >= Y_V_MIN) & (v_g <= Y_V_MAX)
    )

    white_xyz  = np.column_stack((xs_g[is_white_g],
                                  ys_g[is_white_g],
                                  zs_g[is_white_g])).astype(np.float32)
    yellow_xyz = np.column_stack((xs_g[is_yellow_g],
                                  ys_g[is_yellow_g],
                                  zs_g[is_yellow_g])).astype(np.float32)
    
    if LOG: log.info(
        f"Pre-voxel: white:{len(white_xyz)}  "
        f"yellow:{len(yellow_xyz)}  "
        f"not-black:{len(not_black_xyz)}  "
        f"mid:{len(mid_xyz)}"
    )


    # -------- 6. voxel filter all four clouds -----------------------------
    white_xyz      = _voxel_downsample(white_xyz,      VOXEL_SIZE)
    yellow_xyz     = _voxel_downsample(yellow_xyz,     VOXEL_SIZE)
    not_black_xyz  = _voxel_downsample(not_black_xyz,  VOXEL_SIZE)
    mid_xyz        = _voxel_downsample(mid_xyz,        VOXEL_SIZE)

    if LOG: log.info(
        f"Post-voxel: white:{len(white_xyz)}  "
        f"yellow:{len(yellow_xyz)}  "
        f"not-black:{len(not_black_xyz)}  "
        f"mid:{len(mid_xyz)}"
    )

    return white_xyz, yellow_xyz, not_black_xyz, mid_xyz



class PointCloudDownscaler(Node):
    def __init__(self):
        super().__init__("pointcloud_downscaler")
        #self.create_subscription(PointCloud2, "/zed/zed_node/rgb_raw/image_raw_color", self.pointcloud_cb, 10)
        self.create_subscription(PointCloud2, "/camera/points", self.pointcloud_cb, 10)
        self.white_publisher = self.create_publisher(PointCloud2, "/igvc/white_points", 10)
        self.yellow_publisher = self.create_publisher(PointCloud2, "/igvc/yellow_points", 10)
        self.notblack_publisher = self.create_publisher(PointCloud2, "/igvc/notblack_points", 10)
        self.midz_publisher = self.create_publisher(PointCloud2, "/igvc/midz_points", 10)

        self.clouds = None

    def pointcloud_cb(self, msg: PointCloud2):
        white_xyz, yellow_xyz, not_black_xyz, midz_xyz = fast_xyz_white_yellow(msg, self.get_logger())
        self.clouds = (pc2.create_cloud_xyz32(msg.header, white_xyz), pc2.create_cloud_xyz32(msg.header, yellow_xyz), 
                        pc2.create_cloud_xyz32(msg.header, not_black_xyz), pc2.create_cloud_xyz32(msg.header, midz_xyz))
        self.white_publisher.publish(self.clouds[0])
        self.yellow_publisher.publish(self.clouds[1])
        self.notblack_publisher.publish(self.clouds[2])
        self.midz_publisher.publish(self.clouds[3])

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudDownscaler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down…")
    finally:
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



