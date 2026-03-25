import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np

from HAZARD.utils.distributions import Categorical, DiagGaussian
from HAZARD.utils.model_utils import Flatten, NNBase

class Goal_Oriented_Semantic_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 num_sem_categories=16):
        super(Goal_Oriented_Semantic_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)

        out_size = int(input_shape[1] / 16.) * int(input_shape[2] / 16.)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 8, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.linear1 = nn.Linear(out_size * 32 + 8 * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        x = self.main(inputs)
        orientation_emb = self.orientation_emb(extras[:, 0])
        goal_emb = self.goal_emb(extras[:, 1])

        x = torch.cat((x, orientation_emb, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 1:
            self.network = Goal_Oriented_Semantic_Policy(
                obs_shape, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


"""
Same utility functions as in TDWUtils, but:
- use torch instead of numpy
- support batched inputs
"""

class PointCloud:
    __WIDTH: int = -1
    __HEIGHT: int = -1
    @staticmethod
    def get(depth:torch.Tensor, camera_matrix: torch.Tensor, vfov: float = 120.0, near_plane: float = 0.1, far_plane: float = 100, device = torch.device("cpu")) -> torch.Tensor:
        if isinstance(camera_matrix, tuple):
            camera_matrix = torch.Tensor(camera_matrix)
        if len(depth.shape) == 2:
            depth = depth.unsqueeze(0)
        if len(camera_matrix.shape) == 2:
            camera_matrix = camera_matrix.unsqueeze(0)
        camera_matrix = torch.linalg.inv(camera_matrix.reshape((4, 4)))

        # Different from real-world camera coordinate system.
        # OpenGL uses negative z axis as the camera front direction.
        # x axes are same, hence y axis is reversed as well.
        # Source: https://learnopengl.com/Getting-started/Camera
        rot = torch.Tensor([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]]).to(device)
        camera_matrix = torch.matmul(camera_matrix, rot)

        bs, H, W = depth.shape
        # Cache some calculations we'll need to use every time.
        if PointCloud.__HEIGHT != H or PointCloud.__WIDTH != W:
            PointCloud.__HEIGHT = H
            PointCloud.__WIDTH = W

            img_pixs = np.mgrid[0: H, 0: W].reshape(2, -1)
            # Swap (v, u) into (u, v).
            img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
            img_pix_ones = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))

            # Calculate the intrinsic matrix from vertical_fov.
            # Motice that hfov and vfov are different if height != width
            # We can also get the intrinsic matrix from opengl's perspective matrix.
            # http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
            vfov = vfov / 180.0 * np.pi
            tan_half_vfov = np.tan(vfov / 2.0)
            tan_half_hfov = tan_half_vfov * PointCloud.__WIDTH / float(PointCloud.__HEIGHT)
            fx = PointCloud.__WIDTH / 2.0 / tan_half_hfov  # focal length in pixel space
            fy = PointCloud.__HEIGHT / 2.0 / tan_half_vfov
            intrinsics = np.array([[fx, 0, PointCloud.__WIDTH / 2.0],
                                    [0, fy, PointCloud.__HEIGHT / 2.0],
                                    [0, 0, 1]])
            img_inv = np.linalg.inv(intrinsics[:3, :3])
            PointCloud.__CAM_TO_IMG_MAT = np.dot(img_inv, img_pix_ones)
            # PointCloud.__CAM_TO_IMG_MAT = PointCloud.__CAM_TO_IMG_MAT.reshape((1, ) + PointCloud.__CAM_TO_IMG_MAT.shape)

        if isinstance(PointCloud.__CAM_TO_IMG_MAT, np.ndarray):
            PointCloud.__CAM_TO_IMG_MAT = torch.from_numpy(PointCloud.__CAM_TO_IMG_MAT).float().to(device)
        if PointCloud.__CAM_TO_IMG_MAT.device != device:
            PointCloud.__CAM_TO_IMG_MAT = PointCloud.__CAM_TO_IMG_MAT.to(device)

        points_in_cam = torch.multiply(PointCloud.__CAM_TO_IMG_MAT.reshape(3, H * W), depth.reshape(bs, 1, -1))
        points_in_cam = torch.cat((points_in_cam, torch.ones((bs, 1) + points_in_cam.shape[2:], device=device)), dim=1)
        points_in_world = torch.matmul(camera_matrix, points_in_cam)
        points_in_world = points_in_world[:, :3, :].reshape(bs, 3, H, W)
        points_in_cam = points_in_cam[:, :3, :].reshape(bs, 3, H, W)
        return points_in_world

"""
obs: (bs, c, h, w); channels are R, G, B, D, other channels
map: (bs, map_size_h, map_size_v); channels are exp, all, id

be careful. map_size_h and map_size_v are not the real map sizes but the number of grids.
"""


class Semantic_Mapping(nn.Module):
    heights = [0.0, 0.5, 1.0, 1.7]
    def __init__(self, device=None, screen_size=128, map_size_h=64, map_size_v=64, grid_size=0.25, origin_pos=[0.0, 0.0]):
        super().__init__()
        # self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.device = torch.device("cpu")
        self.screen_w = screen_size
        self.screen_h = screen_size
        self.map_size_h = map_size_h
        self.map_size_v = map_size_v
        self.grid_size = grid_size
        self.map_offset: List[int] = [self.map_size_h // 2, self.map_size_v // 2] # the position of (0, 0)
        self.map_offset[0] -= int(origin_pos[0] / self.grid_size + 0.5)
        self.map_offset[1] -= int(origin_pos[1] / self.grid_size + 0.5)
        self.origin_pos = origin_pos
    
    def grid_to_real(self, grid_pos):
        if not isinstance(grid_pos, list):
            grid_pos = grid_pos.tolist()
        return [(grid_pos[0] - self.map_offset[0]) * self.grid_size, 0, (grid_pos[1] - self.map_offset[1]) * self.grid_size]
    def real_to_grid(self, real_pos):
        if not isinstance(real_pos, list):
            real_pos = real_pos.tolist()
        if len(real_pos) > 2:
            real_pos = [real_pos[0], real_pos[2]]
        return [int(real_pos[0] / self.grid_size + self.map_offset[0] + 0.5), int(real_pos[1] / self.grid_size + self.map_offset[1] + 0.5)]
    
    def forward(self, obs, id_map, camera_matrix, maps_last, position: Optional[np.ndarray]=None,
                targets: Optional[List[int]]=None, record_mode=False):
        """ batched input on default """
        if not isinstance(camera_matrix, torch.Tensor):
            camera_matrix = torch.Tensor(camera_matrix).to(self.device)
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)
        if not isinstance(id_map, torch.Tensor):
            id_map = torch.tensor(id_map, dtype=torch.int64, device=self.device)
        
        if maps_last is None:
            maps_last = dict()
            maps_last['explored'] = torch.zeros((self.map_size_h, self.map_size_v), device=self.device, dtype=torch.int64)
            maps_last['height'] = torch.zeros((self.map_size_h, self.map_size_v), device=self.device)
            maps_last['id'] = torch.zeros((self.map_size_h, self.map_size_v), device=self.device, dtype=torch.int64)
            if obs.shape[0] > 4:
                maps_last['other'] = torch.zeros((obs.shape[0] - 4, self.map_size_h, self.map_size_v), device=self.device)
        elif not isinstance(maps_last["explored"], torch.Tensor):
            nmap = dict()
            nmap["explored"] = torch.tensor(maps_last["explored"], device=self.device, dtype=torch.int64)
            nmap["height"] = torch.tensor(maps_last["height"], device=self.device)
            nmap["id"] = torch.tensor(maps_last["id"], device=self.device, dtype=torch.int64)
            if maps_last["other"] is not None:
                nmap["other"] = torch.Tensor(maps_last["other"]).to(self.device)
            maps_last = nmap

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        bs, c, h, w = obs.shape
        depth = obs[:, 3, :, :]
        # depth = torch.min(depth, torch.Tensor([self.map_size_h * self.grid_size * 2]).to(self.device))

        point_cloud = PointCloud.get(depth=depth, camera_matrix=camera_matrix, device=self.device)
        # fout = open("point_cloud.txt", "w")
        # for i in range(bs):
        #     for j in range(h):
        #         for k in range(w):
        #             for t in range(3):
        #                 print(point_cloud[i, t, j, k].item(), end=' ', file=fout)
        #             print('', file=fout)
        # point cloud to map
        Y = point_cloud[0, 1, :, :]
        XZ = ((point_cloud[0, [0, 2], :, :] + self.grid_size * 0.5) // self.grid_size + torch.Tensor(self.map_offset).to(self.device).reshape((2, 1, 1)))
        bound_low = torch.Tensor([0, 0]).to(self.device).reshape((2, 1, 1))
        bound_high = torch.Tensor([self.map_size_h-1, self.map_size_v-1]).to(self.device).reshape((2, 1, 1))
        XZ = torch.max(torch.min(XZ, bound_high), bound_low)
        XZ = XZ.long()
        zipped = XZ[0, :, :] * self.map_size_v + XZ[1, :, :]
        
        zipped = zipped[Y < 2.5]
        id_map = id_map[Y < 2.5]
        Y = Y[Y < 2.5]
        
        sort_Y, order = torch.sort(Y.flatten())
        sort_idx = zipped.flatten()[order]

        map_exp = torch.zeros((self.map_size_h * self.map_size_v), device=self.device, dtype=torch.int64)
        map_height = torch.zeros((self.map_size_h * self.map_size_v), device=self.device)
        map_id  = torch.zeros((self.map_size_h * self.map_size_v), device=self.device, dtype=torch.int64)
        if c > 4:
            map_other = torch.zeros((c - 4, self.map_size_h * self.map_size_v)).to(self.device)
        else:
            map_other = None
        
        map_exp[sort_idx] = 1
        map_height[sort_idx] = sort_Y
        if c > 4:
            for i in range(c-4):
                map_other[i, sort_idx] = obs[0, i+4, :, :].flatten()[order] * sort_Y
        id_map = id_map.flatten()[order]
        
        # this will have to sort according to targets
        map_id[sort_idx[id_map > 0]] = id_map[id_map > 0]
        if targets is not None:
            for target in targets:
                if (id_map == target).any() == 0 and (maps_last["id"] == target).any():
                    map_id[maps_last["id"].flatten() == target] = target
            for target in targets:
                map_id[sort_idx[id_map == target]] = target
        
        map_exp = map_exp.reshape((self.map_size_h, self.map_size_v))
        map_height = map_height.reshape((self.map_size_h, self.map_size_v))
        map_id = map_id.reshape((self.map_size_h, self.map_size_v))
        if c > 4:
            map_other = map_other.reshape((c - 4, self.map_size_h, self.map_size_v))
        
        if position is not None:
            x, z = self.real_to_grid(position)
            R = max(1, int(0.2 // self.grid_size))
            if not record_mode:
                map_exp[max(0, x - R):min(self.map_size_h, x + R + 1),
                max(0, z - R):min(self.map_size_v, z + R + 1)] = 1
                map_height[max(0, x - R):min(self.map_size_h, x + R + 1),
                max(0, z - R):min(self.map_size_v, z + R + 1)] = 0

        map_height = map_height * map_exp + maps_last["height"] * (1 - map_exp)
        map_id  = map_id * map_exp + maps_last["id"] * (1 - map_exp)
        if c > 4:
            map_other = map_other * map_exp + maps_last["other"] * (1 - map_exp)
        map_exp = torch.max(map_exp, maps_last["explored"])
        
        return dict(height=map_height, explored=map_exp, id=map_id, other=map_other)

if __name__ == "__main__":
    pass