# import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
# import vision_module_msgs.srv as vm_srv
from vision_module_msgs.srv import SelectCamera

from .bbox_projection import bbox_image_fraction


class CameraSelectorNode(Node):
    def __init__(self):
        super().__init__("camera_selector_node")

        self.declare_parameter(
            "camera_names",
            ["wrist_zed", "scene_zed_1", "scene_zed_2"],
        )
        self.camera_names = self.get_parameter("camera_names").value

        self.callback_group = ReentrantCallbackGroup()

        self.latest_images: dict[str, Image] = {}
        self.latest_poses: dict[str, PoseStamped] = {}
        self.intrinsics: dict[str, CameraInfo] = {}

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
            image_topic = f"/{camera_name}/zed_node/rgb/color/rect/image"
            pose_topic = f"/{camera_name}/zed_node/pose"
            camera_info_topic = f"/{camera_name}/zed_node/rgb/color/rect/image/camera_info"

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

    def _on_image(self, camera_name: str, msg: Image):
        self.latest_images[camera_name] = msg

    def _on_pose(self, camera_name: str, msg: PoseStamped):
        self.latest_poses[camera_name] = msg

    def _on_camera_info(self, camera_name: str, msg: CameraInfo):
        self.intrinsics[camera_name] = msg

    def _get_hardcoded_object_data(self):
        """Возвращает захардкоженные данные bbox объекта.

        Типы данных специально оставлены такими же, какие дальше ожидает
        bbox_image_fraction():

        center: dict[str, float]
        bbox_sizes: dict[str, float]
        rotation: dict[str, float]

        Важно:
        center должен быть в той же системе координат, что и pose камер.
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

        for camera_name in self.camera_names:
            image_msg = self.latest_images.get(camera_name)
            # pose_msg = self.latest_poses.get(camera_name)
            camera_info = self.intrinsics.get(camera_name)

            if image_msg is None or camera_info is None:
                self.get_logger().warning(
                    f"Camera '{camera_name}' missing data, skipping"
                )
                continue
            if camera_name == 'cam0':
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
                f"Camera '{camera_name}': {str(score)}"
            )

            if score > best_score:
                best_score = score
                best_camera = camera_name

        if best_camera is None:
            response.success = False
            response.message = "No cameras have all required data available"
            response.object_uuid = object_uuid
            response.camera_name = ""
            response.image_topic = ""
            return response

        response.success = True
        response.message = f"Camera selected, score={best_score:.4f}"
        response.object_uuid = object_uuid
        response.camera_name = best_camera
        response.image_topic = f"/{best_camera}/zed_node/rgb/image_rect_color"
        response.rgb_image = self.latest_images[best_camera]
        response.camera_info = self.intrinsics[best_camera]

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