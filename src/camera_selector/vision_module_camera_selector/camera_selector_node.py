# import threading
import os
from datetime import datetime

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
# import vision_module_msgs.srv as vm_srv
from vision_module_msgs.srv import SelectCamera

from .bbox_projection import (
    bbox_image_fraction,
    clip_polygon_to_image,
    convex_hull_2d,
    quaternion_to_rotation_matrix,
)


_BBOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _clip_corners_to_near_plane(camera_corners, min_depth):
    valid = [corner for corner in camera_corners if corner[2] > min_depth]

    for i, j in _BBOX_EDGES:
        zi = camera_corners[i, 2]
        zj = camera_corners[j, 2]

        if (zi > min_depth) != (zj > min_depth):
            t = (min_depth - zi) / (zj - zi)
            valid.append(
                camera_corners[i] + t * (camera_corners[j] - camera_corners[i])
            )

    if not valid:
        return np.empty((0, 3), dtype=np.float64)

    return np.array(valid, dtype=np.float64)


def bbox_image_polygon(
    bbox_sizes,
    center,
    rotation,
    image_width,
    image_height,
    intrinsics,
    camera_position,
    camera_rotation,
    min_depth=0.05,
):
    """Возвращает 2D-полигон проекции bbox на изображение.

    Возвращает:
        np.ndarray shape=(N, 2), где каждая строка — точка [u, v] в пикселях.

    Если объект не виден или находится за камерой, возвращает пустой массив.
    """

    if image_width <= 0 or image_height <= 0:
        return np.empty((0, 2), dtype=np.float64)

    half_x = bbox_sizes["x"] / 2.0
    half_y = bbox_sizes["y"] / 2.0
    half_z = bbox_sizes["z"] / 2.0

    local_corners = np.array(
        [
            [x, y, z]
            for x in (-half_x, half_x)
            for y in (-half_y, half_y)
            for z in (-half_z, half_z)
        ],
        dtype=np.float64,
    )

    object_rotation = quaternion_to_rotation_matrix(
        rotation["qx"],
        rotation["qy"],
        rotation["qz"],
        rotation["qw"],
    )

    object_center = np.array(
        [
            center["x"],
            center["y"],
            center["z"],
        ],
        dtype=np.float64,
    )

    global_corners = (object_rotation @ local_corners.T).T + object_center

    camera_rotation_matrix = quaternion_to_rotation_matrix(
        camera_rotation["qx"],
        camera_rotation["qy"],
        camera_rotation["qz"],
        camera_rotation["qw"],
    )

    camera_translation = np.array(
        [
            camera_position["x"],
            camera_position["y"],
            camera_position["z"],
        ],
        dtype=np.float64,
    )

    camera_corners = (
        camera_rotation_matrix.T
        @ (global_corners - camera_translation).T
    ).T

    clipped_corners = _clip_corners_to_near_plane(
        camera_corners,
        min_depth,
    )

    if len(clipped_corners) < 3:
        return np.empty((0, 2), dtype=np.float64)

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    projected_points = np.column_stack(
        [
            fx * clipped_corners[:, 0] / clipped_corners[:, 2] + cx,
            fy * clipped_corners[:, 1] / clipped_corners[:, 2] + cy,
        ]
    )

    hull = convex_hull_2d(projected_points)

    if len(hull) < 3:
        return np.empty((0, 2), dtype=np.float64)

    clipped_hull = clip_polygon_to_image(
        hull,
        image_width=image_width,
        image_height=image_height,
    )

    if len(clipped_hull) < 3:
        return np.empty((0, 2), dtype=np.float64)

    return clipped_hull


class CameraSelectorNode(Node):
    def __init__(self):
        super().__init__("camera_selector_node")

        self.declare_parameter(
            "camera_names",
            ["cam0", "cam1"],
        )
        self.camera_names = self.get_parameter("camera_names").value

        self.declare_parameter("debug_image_dir", "camera_selector_debug")
        self.debug_image_dir = self.get_parameter("debug_image_dir").value
        os.makedirs(self.debug_image_dir, exist_ok=True)

        self.callback_group = ReentrantCallbackGroup()

        self.latest_images: dict[str, Image] = {}
        self.latest_poses: dict[str, PoseStamped] = {}
        self.intrinsics: dict[str, CameraInfo] = {}

        self.bridge = CvBridge()

        # Старый вариант: клиент к scene graph.
        # Больше не нужен, потому что bbox объекта теперь захардкожен.
        #
        # self.object_registry_client = self.create_client(
        #     vm_srv.GetSceneObject,
        #     "/scene_graph/get_object",
        #     callback_group=self.callback_group
        # )

        self.select_camera_service = self.create_service(
            # vm_srv.SelectCamera,
            SelectCamera,
            "/camera_selector/select_camera",
            self._select_camera_callback,
            callback_group=self.callback_group,
        )

        self._subscribe_to_cameras()

    def _subscribe_to_cameras(self):
        for camera_name in self.camera_names:
            image_topic = f"/{camera_name}/zed_node/left/color/rect/image"
            # pose_topic = f"/{camera_name}/zed_node/pose"
            camera_info_topic = (
                f"/{camera_name}/zed_node/left/color/rect/image/camera_info"
            )

            self.create_subscription(
                Image,
                image_topic,
                lambda msg, name=camera_name: self._on_image(name, msg),
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )

            # self.create_subscription(
            #     PoseStamped,
            #     pose_topic,
            #     lambda msg, name=camera_name: self._on_pose(name, msg),
            #     qos_profile_sensor_data,
            #     callback_group=self.callback_group,
            # )

            self.create_subscription(
                CameraInfo,
                camera_info_topic,
                lambda msg, name=camera_name: self._on_camera_info(name, msg),
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )

            self.get_logger().info(f"Subscribed to camera '{camera_name}'")
            self.get_logger().info(f"  image_topic: {image_topic}")
            self.get_logger().info(f"  camera_info_topic: {camera_info_topic}")

    def _on_image(self, camera_name: str, msg: Image):
        self.latest_images[camera_name] = msg

    def _on_pose(self, camera_name: str, msg: PoseStamped):
        self.latest_poses[camera_name] = msg

    def _on_camera_info(self, camera_name: str, msg: CameraInfo):
        self.intrinsics[camera_name] = msg

    def _get_hardcoded_object_data(self):
        """Возвращает захардкоженные данные bbox объекта.

        Типы данных:

        center: dict[str, float]
        bbox_sizes: dict[str, float]
        rotation: dict[str, float]

        Важно:
        center должен быть в той же системе координат, что и extrinsics камер.
        rotation — quaternion в формате x, y, z, w.
        """

        center: dict[str, float] = {
            "x": -0.28,
            "y": 0.26,
            "z": 0.05,
        }

        bbox_sizes: dict[str, float] = {
            "x": 0.15,
            "y": 0.18,
            "z": 0.09,
        }

        rotation: dict[str, float] = {
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }

        return center, bbox_sizes, rotation

    def _get_hardcoded_camera_pose(self, camera_name: str):
        """Возвращает захардкоженные extrinsics камеры.

        camera_position:
            положение камеры в общей системе координат.

        camera_rotation:
            ориентация камеры в общей системе координат.
        """

        if camera_name == "cam0":
            camera_position: dict[str, float] = {
                "x": 0.54866,
                "y": 0.2903,
                "z": 0.5205,
            }

            camera_rotation: dict[str, float] = {
                "qx": 0.023,
                "qy": -0.999,
                "qz": 0.0296,
                "qw": 0.003,
            }
        else:
            camera_position: dict[str, float] = {
                "x": 0.182,
                "y": 0.278,
                "z": 0.516,
            }

            camera_rotation: dict[str, float] = {
                "qx": -0.0005,
                "qy": -0.999,
                "qz": 0.007,
                "qw": 0.0003,
            }

        return camera_position, camera_rotation

    # Старый вариант: получение объекта через /scene_graph/get_object.
    # Полностью закомментирован, потому что теперь не нужен.
    #
    # def _get_scene_object(self, object_uuid: str):
    #     if not self.object_registry_client.wait_for_service(timeout_sec=1.0):
    #         self.get_logger().error("object_registry service not available")
    #         return None
    #
    #     request = vm_srv.GetSceneObject.Request()
    #     request.object_uuid = object_uuid
    #
    #     future = self.object_registry_client.call_async(request)
    #
    #     done_event = threading.Event()
    #     future.add_done_callback(lambda _: done_event.set())
    #     done_event.wait(timeout=5.0)
    #
    #     if not done_event.is_set():
    #         self.object_registry_client.remove_pending_request(future)
    #         self.get_logger().error(f"Timeout getting object {object_uuid}")
    #         return None
    #
    #     return future.result()

    def _save_selected_image_with_bbox(
        self,
        object_uuid: str,
        camera_name: str,
        image_msg: Image,
        bbox_sizes: dict[str, float],
        center: dict[str, float],
        rotation: dict[str, float],
        intrinsics: dict[str, float],
        camera_position: dict[str, float],
        camera_rotation: dict[str, float],
    ) -> str | None:
        """Сохраняет выбранное изображение с красной рамкой bbox."""

        polygon = bbox_image_polygon(
            bbox_sizes=bbox_sizes,
            center=center,
            rotation=rotation,
            image_width=image_msg.width,
            image_height=image_msg.height,
            intrinsics=intrinsics,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
        )

        if len(polygon) < 3:
            self.get_logger().warning(
                "Object projection is empty, debug image was not saved"
            )
            return None

        cv_image = self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding="bgr8",
        )

        points = np.round(polygon).astype(np.int32)
        points = points.reshape((-1, 1, 2))

        red_bgr = (0, 0, 255)
        thickness = 3

        cv2.polylines(
            cv_image,
            [points],
            isClosed=True,
            color=red_bgr,
            thickness=thickness,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_object_uuid = object_uuid.replace("/", "_").replace(" ", "_")

        filename = f"{timestamp}_{safe_object_uuid}_{camera_name}.png"
        output_path = os.path.join(self.debug_image_dir, filename)

        success = cv2.imwrite(output_path, cv_image)

        if not success:
            self.get_logger().error(
                f"Failed to save debug image: {output_path}"
            )
            return None

        self.get_logger().info(f"Saved debug image: {output_path}")

        return output_path

    def _select_camera_callback(self, request, response):
        object_uuid = request.object_uuid

        # Старый вариант: получали объект из scene graph.
        #
        # object_response = self._get_scene_object(object_uuid)
        #
        # if object_response is None or not object_response.success:
        #     response.success = False
        #     response.message = f"Object {object_uuid} not found in registry"
        #     response.object_uuid = object_uuid
        #     return response
        #
        # scene_object = object_response.object
        #
        # center = {
        #     "x": scene_object.bbox.center.position.x,
        #     "y": scene_object.bbox.center.position.y,
        #     "z": scene_object.bbox.center.position.z,
        # }
        # bbox_sizes = {
        #     "x": scene_object.bbox.size.x,
        #     "y": scene_object.bbox.size.y,
        #     "z": scene_object.bbox.size.z,
        # }
        # rotation = {
        #     "qx": scene_object.bbox.center.orientation.x,
        #     "qy": scene_object.bbox.center.orientation.y,
        #     "qz": scene_object.bbox.center.orientation.z,
        #     "qw": scene_object.bbox.center.orientation.w,
        # }

        # Новый вариант: bbox объекта не запрашивается,
        # а берётся из захардкоженных данных.
        center, bbox_sizes, rotation = self._get_hardcoded_object_data()

        best_camera = None
        best_score = -1.0
        best_camera_position = None
        best_camera_rotation = None
        best_intrinsics = None

        for camera_name in self.camera_names:
            image_msg = self.latest_images.get(camera_name)
            # pose_msg = self.latest_poses.get(camera_name)
            camera_info = self.intrinsics.get(camera_name)

            if image_msg is None or camera_info is None:
                self.get_logger().warning(
                    f"Camera '{camera_name}' missing data, skipping"
                )
                continue

            camera_position, camera_rotation = self._get_hardcoded_camera_pose(
                camera_name
            )

            intrinsics: dict[str, float] = {
                "fx": camera_info.k[0],
                "fy": camera_info.k[4],
                "cx": camera_info.k[2],
                "cy": camera_info.k[5],
            }

            score = bbox_image_fraction(
                bbox_sizes=bbox_sizes,
                center=center,
                rotation=rotation,
                image_width=image_msg.width,
                image_height=image_msg.height,
                intrinsics=intrinsics,
                camera_position=camera_position,
                camera_rotation=camera_rotation,
            )

            self.get_logger().warning(
                f"Camera '{camera_name}': {score}"
            )

            if score > best_score:
                best_score = score
                best_camera = camera_name
                best_camera_position = camera_position
                best_camera_rotation = camera_rotation
                best_intrinsics = intrinsics

        if best_camera is None:
            response.success = False
            response.message = "No cameras have all required data available"
            response.object_uuid = object_uuid
            response.camera_name = ""
            response.image_topic = ""
            return response

        response.success = True
        response.object_uuid = object_uuid
        response.camera_name = best_camera
        response.image_topic = f"/{best_camera}/zed_node/rgb/color/rect/image"
        response.rgb_image = self.latest_images[best_camera]
        response.camera_info = self.intrinsics[best_camera]

        saved_image_path = None

        try:
            saved_image_path = self._save_selected_image_with_bbox(
                object_uuid=object_uuid,
                camera_name=best_camera,
                image_msg=self.latest_images[best_camera],
                bbox_sizes=bbox_sizes,
                center=center,
                rotation=rotation,
                intrinsics=best_intrinsics,
                camera_position=best_camera_position,
                camera_rotation=best_camera_rotation,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to save debug image: {exc}")

        if saved_image_path is None:
            response.message = (
                f"Camera selected, score={best_score:.4f}; "
                "debug image was not saved"
            )
        else:
            response.message = (
                f"Camera selected, score={best_score:.4f}; "
                f"debug image: {saved_image_path}"
            )

        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraSelectorNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()