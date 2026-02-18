"""Vendored EB-Manipulation utilities for EASI.

Adapted from EmbodiedBench/embodiedbench/envs/eb_manipulation/eb_man_utils.py.
Changes:
- Constants (SCENE_BOUNDS, ROTATION_RESOLUTION, VOXEL_SIZE) made into function
  parameters with defaults matching EmbodiedBench.
- YOLO model loaded lazily (not at module import time).
- All functions preserved for full benchmark alignment.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

# Default constants (configurable via function parameters)
DEFAULT_SCENE_BOUNDS = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
DEFAULT_ROTATION_RESOLUTION = 3
DEFAULT_VOXEL_SIZE = 100
CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
USE_GENERAL_OBJECT_NAMES = True

# Lazy-loaded YOLO model
_yolo_model = None


def _get_yolo_model():
    """Lazy-load YOLO model on first use."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        _yolo_model = YOLO("yolo11n.pt")
    return _yolo_model


# From https://github.com/stepjam/RLBench/blob/master/rlbench/backend/utils.py
def point_to_voxel_index(
    point: np.ndarray,
    scene_bounds: np.ndarray = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
):
    if scene_bounds is None:
        scene_bounds = DEFAULT_SCENE_BOUNDS
    bb_mins = np.array(scene_bounds[0:3])[None]
    bb_maxs = np.array(scene_bounds[3:])[None]
    dims_m_one = np.array([voxel_size] * 3)[None] - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one
    )

    return voxel_indicy.reshape(point.shape)


def discrete_euler_to_quaternion(
    discrete_euler,
    rotation_resolution: int = DEFAULT_ROTATION_RESOLUTION,
):
    from scipy.spatial.transform import Rotation

    euluer = (discrete_euler * rotation_resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


def get_continuous_action_from_discrete(
    discrete_action,
    scene_bounds: np.ndarray = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
    rotation_resolution: int = DEFAULT_ROTATION_RESOLUTION,
):
    if scene_bounds is None:
        scene_bounds = DEFAULT_SCENE_BOUNDS
    trans_indicies = np.array(discrete_action[:3])
    rot_and_grip_indicies = np.array(discrete_action[3:6])
    is_gripper_open = discrete_action[6]

    bounds = scene_bounds
    res = (bounds[3:] - bounds[:3]) / voxel_size
    attention_coordinate = bounds[:3] + res * trans_indicies + res / 2
    quat = discrete_euler_to_quaternion(rot_and_grip_indicies, rotation_resolution)

    continuous_action = np.concatenate(
        [attention_coordinate, quat, [is_gripper_open]]
    )
    return continuous_action


def draw_xyz_coordinate(image_path, resolution):
    import cv2

    image = cv2.imread(image_path)
    if resolution == 500:
        origin = (62, 239)
        axis_length = 30
        color_x = (0, 0, 255)
        color_y = (0, 255, 0)
        color_z = (255, 0, 0)

        cv2.circle(image, (int(origin[0]), int(origin[1])), 3, (0, 0, 255), -1)
        cv2.putText(
            image,
            "(0, 0)",
            (62, 255),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.3,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.arrowedLine(
            image,
            origin,
            (origin[0] + axis_length, origin[1]),
            color_y,
            2,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "y",
            (origin[0] + axis_length, origin[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_y,
            2,
        )
        cv2.arrowedLine(
            image,
            origin,
            (origin[0], origin[1] - axis_length),
            color_z,
            2,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "z",
            (origin[0] - 20, origin[1] - axis_length),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_z,
            2,
        )
        cv2.arrowedLine(
            image,
            origin,
            (origin[0] - axis_length + 12, origin[1] + axis_length),
            color_x,
            2,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "x",
            (origin[0] - axis_length, origin[1] + axis_length - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_x,
            2,
        )
        cv2.imwrite(image_path, image)

    elif resolution == 300:
        origin = (38, 142)
        axis_length = 30
        color_x = (0, 0, 255)
        color_y = (0, 255, 0)
        color_z = (255, 0, 0)

        cv2.circle(image, (int(origin[0]), int(origin[1])), 3, (0, 0, 255), -1)
        cv2.putText(
            image,
            "(0, 0)",
            (38, 158),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.3,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.arrowedLine(
            image,
            origin,
            (origin[0] + axis_length, origin[1]),
            color_y,
            2,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "y",
            (origin[0] + axis_length, origin[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_y,
            2,
        )
        cv2.arrowedLine(
            image,
            origin,
            (origin[0], origin[1] - axis_length),
            color_z,
            2,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "z",
            (origin[0] - 20, origin[1] - axis_length),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_z,
            2,
        )
        cv2.arrowedLine(
            image,
            origin,
            (origin[0] - axis_length + 12, origin[1] + axis_length),
            color_x,
            2,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "x",
            (origin[0] - axis_length, origin[1] + axis_length - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_x,
            2,
        )
        cv2.imwrite(image_path, image)

    elif resolution == 700:
        origin = (88, 335)
        axis_length = 50
        color_x = (0, 0, 255)
        color_y = (0, 255, 0)
        color_z = (255, 0, 0)

        cv2.circle(image, (int(origin[0]), int(origin[1])), 3, (0, 0, 255), -1)
        cv2.putText(
            image,
            "(0, 0)",
            (88, 355),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.3,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.arrowedLine(
            image,
            origin,
            (origin[0] + axis_length, origin[1]),
            color_y,
            2,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "y",
            (origin[0] + axis_length, origin[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_y,
            2,
        )
        cv2.arrowedLine(
            image,
            origin,
            (origin[0], origin[1] - axis_length),
            color_z,
            2,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "z",
            (origin[0] - 20, origin[1] - axis_length),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_z,
            2,
        )
        cv2.arrowedLine(
            image,
            origin,
            (origin[0] - axis_length + 20, origin[1] + axis_length),
            color_x,
            2,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "x",
            (origin[0] - axis_length, origin[1] + axis_length - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_x,
            2,
        )
        cv2.imwrite(image_path, image)
    else:
        raise ValueError(
            "Detection boxes are not supported for this resolution. "
            "Please disable detection boxes or use a valid resolution."
        )

    return image_path


def increase_bbox(bbox, scale_factor=1):
    """Increase the bounding box size by a scale factor."""
    x1, y1, x2, y2 = bbox
    original_width = x2 - x1
    original_height = y2 - y1
    center_x = x1 + original_width / 2
    center_y = y1 + original_height / 2
    new_width = original_width * scale_factor
    new_height = original_height * scale_factor
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)
    return [new_x1, new_y1, new_x2, new_y2]


def draw_bounding_boxes(
    image_path_list, world_points, camera_extrinsics_list, camera_intrinsics_list
):
    import cv2

    model = _get_yolo_model()
    image_save_path_list = []
    for input_image_path, camera_extrinsics, camera_intrinsics in zip(
        image_path_list, camera_extrinsics_list, camera_intrinsics_list
    ):
        T_inv = np.linalg.inv(camera_extrinsics)
        rvec = T_inv[:3, :3]
        tvec = T_inv[:3, 3]
        pixel_points_2D, _ = cv2.projectPoints(
            np.array(world_points), rvec, tvec, camera_intrinsics, np.zeros(4)
        )

        results = model.predict(
            source=input_image_path, conf=0.0001, line_width=1, verbose=False
        )
        predicted_boxes = results[0].boxes.xyxy
        image_bgr = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

        box_id = 0
        text_positions = []
        for point in pixel_points_2D:
            x, y = point[0]
            min_dist = float("inf")
            min_idx = -1
            for i, box in enumerate(predicted_boxes):
                center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
                dist = (center[0] - x) ** 2 + (center[1] - y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            if min_dist > 400:
                continue
            increased_box = increase_bbox(predicted_boxes[min_idx], 1.2)
            center_pixel = (0, 0, 255)
            cv2.rectangle(
                image_bgr,
                (int(increased_box[0]), int(increased_box[1])),
                (int(increased_box[2]), int(increased_box[3])),
                center_pixel,
                1,
            )
            text_position = (int(increased_box[0]) + 20, int(increased_box[1]) - 10)
            for pos in text_positions:
                if abs(pos[0] - text_position[0]) < 10 and abs(
                    pos[1] - text_position[1]
                ) < 10:
                    text_position = (
                        int(increased_box[0]) + 10,
                        int(increased_box[1]) - 10,
                    )
            text_positions.append(text_position)
            cv2.putText(
                image_bgr,
                str(box_id + 1),
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                center_pixel,
                1,
                cv2.LINE_AA,
            )
            box_id += 1
        base, ext = os.path.splitext(input_image_path)
        image_save_path = f"{base}_annotated{ext}"
        cv2.imwrite(image_save_path, image_bgr)
        image_save_path_list.append(image_save_path)

    return image_save_path_list


####### Generate object information for the initial observation
def _get_mask_id_to_name_dict_for_input(object_info):
    mask_id_to_name_dict = {}
    for obj in object_info:
        if "id" in object_info[obj]:
            mask_id_to_name_dict[object_info[obj]["id"]] = obj
    return mask_id_to_name_dict


def _get_point_cloud_dict_for_input(obs, camera_types):
    from pyrep.objects import VisionSensor

    point_cloud_dict = {}
    camera_extrinsics_list, camera_intrinsics_list = [], []
    for camera_type in CAMERAS:
        cam_extrinsics = obs["misc"][f"{camera_type}_camera_extrinsics"]
        cam_intrinsics = obs["misc"][f"{camera_type}_camera_intrinsics"]
        if camera_type + "_rgb" in camera_types:
            camera_extrinsics_list.append(cam_extrinsics)
            camera_intrinsics_list.append(cam_intrinsics)
        cam_depth = obs[f"{camera_type}_depth"]
        near = obs["misc"][f"{camera_type}_camera_near"]
        far = obs["misc"][f"{camera_type}_camera_far"]
        cam_depth = (far - near) * cam_depth + near
        point_cloud_dict[camera_type] = (
            VisionSensor.pointcloud_from_depth_and_camera_params(
                cam_depth, cam_extrinsics, cam_intrinsics
            )
        )

    return point_cloud_dict, camera_extrinsics_list, camera_intrinsics_list


def _get_mask_dict_for_input(obs):
    mask_dict = {}
    for camera in CAMERAS:
        rgb_mask = np.array(obs[f"{camera}_mask"], dtype=int)
        mask_dict[camera] = rgb_mask
    return mask_dict


def form_obs_for_input(
    mask_dict,
    mask_id_to_real_name,
    point_cloud_dict,
    scene_bounds: np.ndarray = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
):
    if scene_bounds is None:
        scene_bounds = DEFAULT_SCENE_BOUNDS

    # convert object id to char and average and discretize point cloud per object
    uniques = np.unique(np.concatenate(list(mask_dict.values()), axis=0))
    real_name_to_avg_coord = {}
    all_avg_point_list = []
    for _, mask_id in enumerate(uniques):
        if mask_id not in mask_id_to_real_name:
            continue
        avg_point_list = []
        for camera in CAMERAS:
            mask = mask_dict[camera]
            point_cloud = point_cloud_dict[camera]
            if not np.any(mask == mask_id):
                continue
            avg_point_list.append(
                np.mean(point_cloud[mask == mask_id].reshape(-1, 3), axis=0)
            )

        avg_point = sum(avg_point_list) / len(avg_point_list)
        all_avg_point_list.append(avg_point)
        real_name = mask_id_to_real_name[mask_id]
        real_name_to_avg_coord[real_name] = list(
            point_to_voxel_index(avg_point, scene_bounds=scene_bounds, voxel_size=voxel_size)
        )
    if USE_GENERAL_OBJECT_NAMES:
        implicit_name_to_avg_coord = {}
        i = 1
        for key, value in real_name_to_avg_coord.items():
            implicit_name_to_avg_coord[f"object {i}"] = value
            i += 1
        real_name_to_avg_coord = implicit_name_to_avg_coord

    # Sort the objects based on the y-coordinate
    sorted_indices = sorted(
        range(len(all_avg_point_list)), key=lambda i: all_avg_point_list[i][1]
    )
    all_avg_point_list = [all_avg_point_list[i] for i in sorted_indices]

    # Sort the objects in the general name based on the same order
    real_name_to_avg_coord = sorted(
        real_name_to_avg_coord.items(), key=lambda item: item[1][1]
    )
    if USE_GENERAL_OBJECT_NAMES:
        real_name_to_avg_coord = {
            f"object {i+1}": value
            for i, (_, value) in enumerate(real_name_to_avg_coord)
        }
    else:
        real_name_to_avg_coord = {
            f"{obj_name}": value for obj_name, value in real_name_to_avg_coord
        }
    return real_name_to_avg_coord, all_avg_point_list


def form_object_coord_for_input(obs, task_class, camera_types):
    mask_id_to_sim_name = _get_mask_id_to_name_dict_for_input(
        obs["object_informations"]
    )
    point_cloud_dict, camera_extrinsics_list, camera_intrinsics_list = (
        _get_point_cloud_dict_for_input(obs, camera_types)
    )
    mask_dict = _get_mask_dict_for_input(obs)

    task_handler = TASK_HANDLERS[task_class]()
    sim_name_to_real_name = task_handler.sim_name_to_real_name
    mask_id_to_real_name = {
        mask_id: sim_name_to_real_name[name]
        for mask_id, name in mask_id_to_sim_name.items()
        if name in sim_name_to_real_name
    }
    avg_coord, all_avg_point_list = form_obs_for_input(
        mask_dict, mask_id_to_real_name, point_cloud_dict
    )
    return avg_coord, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list


class base_task_handler:
    def __init__(self, sim_name_to_real_name):
        self.sim_name_to_real_name = sim_name_to_real_name


class pick_cube_shape(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "small_container0": "first container",
            "small_container1": "second container",
            "star_normal_visual0": "first star",
            "star_normal_visual1": "second star",
            "cylinder_normal0": "first cylinder",
            "cylinder_normal1": "second cylinder",
            "triangular_normal0": "first triangular",
            "triangular_normal1": "second triangular",
            "cube_basic0": "first cube",
            "cube_basic1": "second cube",
            "moon_normal_visual0": "first moon",
            "moon_normal_visual1": "second moon",
        }
        super().__init__(sim_name_to_real_name)


class stack_cubes_color(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "star_normal_visual0": "first star",
            "star_normal_visual1": "second star",
            "star_normal_visual2": "third star",
            "star_normal_visual3": "fourth star",
            "cylinder_normal0": "first cylinder",
            "cylinder_normal1": "second cylinder",
            "cylinder_normal2": "third cylinder",
            "cylinder_normal3": "fourth cylinder",
            "triangular_normal0": "first triangular",
            "triangular_normal1": "second triangular",
            "triangular_normal2": "third triangular",
            "triangular_normal3": "fourth triangular",
            "cube_basic0": "first cube",
            "cube_basic1": "second cube",
            "cube_basic2": "third cube",
            "cube_basic3": "fourth cube",
            "moon_normal_visual0": "first moon",
            "moon_normal_visual1": "second moon",
            "moon_normal_visual2": "third moon",
            "moon_normal_visual3": "fourth moon",
        }
        super().__init__(sim_name_to_real_name)


class place_into_shape_sorter_color(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "star_normal_visual0": "first star",
            "star_normal_visual1": "second star",
            "star_normal_visual2": "third star",
            "star_normal_visual3": "fourth star",
            "cylinder_normal0": "first cylinder",
            "cylinder_normal1": "second cylinder",
            "cylinder_normal2": "third cylinder",
            "cylinder_normal3": "fourth cylinder",
            "triangular_normal0": "first triangular",
            "triangular_normal1": "second triangular",
            "triangular_normal2": "third triangular",
            "triangular_normal3": "fourth triangular",
            "cube_basic0": "first cube",
            "cube_basic1": "second cube",
            "cube_basic2": "third cube",
            "cube_basic3": "fourth cube",
            "moon_normal_visual0": "first moon",
            "moon_normal_visual1": "second moon",
            "moon_normal_visual2": "third moon",
            "moon_normal_visual3": "fourth moon",
            "shape_sorter_visual": "shape sorter",
        }
        super().__init__(sim_name_to_real_name)


class wipe_table_shape(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "rectangle": "first rectangle area",
            "rectangle0": "second rectangle area",
            "round": "first round area",
            "round0": "second round area",
            "triangle": "first triangle area",
            "triangle0": "second triangle area",
            "star": "first star area",
            "star0": "second star area",
            "sponge_visual0": "sponge",
        }
        super().__init__(sim_name_to_real_name)


TASK_HANDLERS = {
    "pick": pick_cube_shape,
    "stack": stack_cubes_color,
    "place": place_into_shape_sorter_color,
    "wipe": wipe_table_shape,
}
