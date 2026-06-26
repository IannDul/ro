import numpy as np


_BBOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quaternion)

    if norm < 1e-10:
        return np.eye(3, dtype=np.float64)

    qx, qy, qz, qw = quaternion / norm

    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _clip_corners_to_near_plane(camera_corners, min_depth):
    valid = [c for c in camera_corners if c[2] > min_depth]

    for i, j in _BBOX_EDGES:
        zi, zj = camera_corners[i, 2], camera_corners[j, 2]
        if (zi > min_depth) != (zj > min_depth):
            t = (min_depth - zi) / (zj - zi)
            valid.append(camera_corners[i] + t * (camera_corners[j] - camera_corners[i]))

    return np.array(valid, dtype=np.float64) if valid else np.empty((0, 3), dtype=np.float64)


def convex_hull_2d(points):
    points = sorted(set(map(tuple, points)))

    if len(points) <= 1:
        return np.array(points, dtype=np.float64)

    def cross(origin, point_a, point_b):
        return (
                (point_a[0] - origin[0]) * (point_b[1] - origin[1])
                - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
        )

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_area(points):
    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def clip_polygon_to_image(points, image_width, image_height):
    polygon = points.tolist()

    def clip_by_edge(polygon, edge_name, edge_value):
        if not polygon:
            return []

        clipped = []

        def is_inside(point):
            x, y = point

            if edge_name == "left":
                return x >= edge_value
            if edge_name == "right":
                return x <= edge_value
            if edge_name == "top":
                return y >= edge_value
            if edge_name == "bottom":
                return y <= edge_value

            return False

        def intersection(start, end):
            x1, y1 = start
            x2, y2 = end

            if edge_name in ("left", "right"):
                denominator = x2 - x1

                if abs(denominator) < 1e-12:
                    return end

                t = (edge_value - x1) / denominator
                return [edge_value, y1 + t * (y2 - y1)]

            denominator = y2 - y1

            if abs(denominator) < 1e-12:
                return end

            t = (edge_value - y1) / denominator
            return [x1 + t * (x2 - x1), edge_value]

        previous = polygon[-1]
        previous_inside = is_inside(previous)

        for current in polygon:
            current_inside = is_inside(current)

            if current_inside:
                if not previous_inside:
                    clipped.append(intersection(previous, current))
                clipped.append(current)
            elif previous_inside:
                clipped.append(intersection(previous, current))

            previous = current
            previous_inside = current_inside

        return clipped

    polygon = clip_by_edge(polygon, "left", 0.0)
    polygon = clip_by_edge(polygon, "right", float(image_width))
    polygon = clip_by_edge(polygon, "top", 0.0)
    polygon = clip_by_edge(polygon, "bottom", float(image_height))

    return np.array(polygon, dtype=np.float64)


def bbox_image_fraction(bbox_sizes, center, rotation,
                        image_width, image_height,
                        intrinsics,
                        camera_position, camera_rotation,
                        min_depth=0.05):
    if image_width <= 0 or image_height <= 0:
        return 0.0

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

    camera_corners = (camera_rotation_matrix.T @ (global_corners - camera_translation).T).T

    clipped_corners = _clip_corners_to_near_plane(camera_corners, min_depth)

    if len(clipped_corners) < 3:
        return 0.0

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
        return 0.0

    clipped_hull = clip_polygon_to_image(
        hull,
        image_width=image_width,
        image_height=image_height,
    )

    projected_area = polygon_area(clipped_hull)
    image_area = float(image_width * image_height)

    fraction = projected_area / image_area

    return float(np.clip(fraction, 0.0, 1.0))
