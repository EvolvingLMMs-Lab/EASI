from tdw.add_ons.log_playback import LogPlayback
import os
import numpy as np
from tdw.tdw_utils import TDWUtils

class SceneSetup:
    def __init__(self, data_dir: str, is_flood = False, record_mode = False) -> None:
        playback = LogPlayback()
        playback.load(os.path.join(data_dir, "log.txt"))
        self.commands_list = playback.playback

        if record_mode:
            new_commands_list = []
            table_ids = []
            obstacles = []
            for commands in self.commands_list:
                new_commands = []
                for command in commands:
                    if command["$type"] == "add_object" and "table" in command["category"]:
                        table_ids.append(command["id"])
                    if command["$type"] == "rotate_object_by" and command["id"] in table_ids:
                        command["angle"] += 90
                        print(command)
                    if command["$type"] == "add_object":
                        for obs in ["table", "shelf", "cabinet", "chair", "stool", "suitcase", "basket", ]:
                            if obs in command["category"] and command["id"] not in obstacles:
                                obstacles.append(command["id"])
                    if command["$type"] == "terminate":
                        break
                    new_commands.append(command)
                new_commands_list.append(new_commands)
            new_commands_list.append([{"$type": "bake_nav_mesh"}])
            new_commands_list.append(
                [{"$type": "make_nav_mesh_obstacle", "id": obs, "carve_type": "stationary"} for obs in obstacles])
            self.commands_list = new_commands_list

        import json
        with open(os.path.join(data_dir, "info.json"), "r") as f:
            info = json.load(f)
            
            self.task = info["task"]
            self.containers = info["containers"]
            self.agent_positions = np.array(info["agent"])
            if len(self.agent_positions.shape) == 1:
                self.agent_positions = self.agent_positions.reshape(1, -1)

            if self.task == "wind":
                self.targets = []
                self.target_ids = info["targets"]
            else:
                self.targets = info["targets"]
                self.target_ids = []
            self.target_names = []
            self.target_id2category = {}
            self.target_id2name = {}
            log_lines = open(os.path.join(data_dir, "log.txt")).readlines()
            log_lines = [json.loads(line) for line in log_lines]
            log_lines = sum(log_lines, [])
            log_objects = [log for log in log_lines if log['$type'] == 'add_object']

            if self.task == "wind":
                for target in self.target_ids:
                    for obj in log_objects:
                        # if target == 14415226 and obj['id'] == 14415226: print(obj['id'], target, obj['category'])
                        if obj['id'] == target and obj['category'] not in self.targets:
                            self.targets.append(obj['category'])
                        if obj['id'] == target and obj['name'] not in self.target_names:
                            self.target_names.append(obj['name'])
                        self.target_id2category[obj['id']] = obj['category']
                        self.target_id2name[obj['id']] = obj['name']
            else:
                for target in self.targets:
                    for obj in log_objects:
                        if obj['category'] == target and obj['id'] not in self.target_ids:
                            self.target_ids.append(obj['id'])
                            self.target_id2category[obj['id']] = obj['category']
                            self.target_id2name[obj['id']] = obj['name']
                        if obj['category'] == target and obj['name'] not in self.target_names:
                            self.target_names.append(obj['name'])

            self.other = info["other"]
            f.close()

        if is_flood:
            from envs.flood.object import ObjectStatus
            with open(os.path.join(data_dir, "flood.json"), "r") as f:
                info = json.load(f)
                self.flood_positions = [np.array(source) for source in info["source"]]
                # [np.array([45, 0, 0])]
                self.flood_directions = [np.array(direction) for direction in info["direction"]]
                self.flood_speed = info["speed"]
                self.flood_source_from = info["flood_source_from"]
                f.close()
            self.objects = []
            self.BUOYANCY_LIST = ["chair", "lamp", "backpack", "basket", "pillow", "bag"]
            for l in self.commands_list:
                for c in l:
                    if c["$type"] == "add_object":
                        name = c["name"]
                        idx = c["id"]
                        pos = TDWUtils.vector3_to_array(c["position"])
                        self.objects.append(ObjectStatus(idx=idx, position=pos,
                                                         has_buoyancy=self.naive_judge_buoyancy(name),
                                                         waterproof=self.naive_judge_waterproof(name)))
                    if c["$type"] == "add_scene":
                        self.scene_name = c["name"]
    
    def naive_judge_waterproof(self, name):
        return False

    def naive_judge_buoyancy(self, name):
        for subname in self.BUOYANCY_LIST:
            if subname in name:
                return True
        return False
