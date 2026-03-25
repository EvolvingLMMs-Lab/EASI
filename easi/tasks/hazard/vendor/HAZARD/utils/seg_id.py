import numpy as np
from tdw.output_data import SegmentationColors
from tdw.output_data import ReplicantSegmentationColors
from typing import Union

import os
PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until it contains envs folder
while not os.path.exists(os.path.join(PATH, "envs")):
    PATH = os.path.dirname(PATH)

class SegmentationID:
    """
    replicant and object ids should all be distinct
    """
    def __init__(self):
        self.segmentation_colors = dict()
        self.names = dict()
        self.categories = dict()
        self.reverse_id = dict()
        self.reverse_id[0] = 0

        category_file = os.path.join(PATH, 'data', 'meta_data', 'categories_new.txt')
        with open(category_file, 'r') as f:
            self.global_categories = eval(f.read())
    
    def process(self, segm: Union[SegmentationColors, ReplicantSegmentationColors], id_renumbering: dict = dict()):
        if isinstance(segm, SegmentationColors):
            for j in range(segm.get_num()):
                idx = segm.get_object_id(j)
                self.segmentation_colors[idx] = np.array(segm.get_object_color(j))
                col = self.segmentation_colors[idx]
                # with open("utils/seg_id.txt", "a") as f:
                #     print("idx:", idx, "col:", col, "name:", segm.get_object_name(j).lower(), "category:", segm.get_object_category(j), file=f)
                #     f.close()
                col = col[0] + col[1] * 256 + col[2] * 256 * 256
                if len(id_renumbering) == 0:
                    self.reverse_id[col] = idx
                else:
                    self.reverse_id[col] = id_renumbering[idx] if idx in id_renumbering else 0
                self.names[idx] = segm.get_object_name(j).lower()
                self.categories[idx] = segm.get_object_category(j)
        elif isinstance(segm, ReplicantSegmentationColors):
            for j in range(segm.get_num()):
                idx = segm.get_id(j)
                self.segmentation_colors[idx] = np.array(segm.get_segmentation_color(j))
                col = self.segmentation_colors[idx]
                # with open("utils/seg_id.txt", "a") as f:
                #     print("idx:", idx, "col:", col, "name:", "replicant", "category:", "replicant", file=f)
                #     f.close()
                # print("col:", col)
                col = col[0] + col[1] * 256 + col[2] * 256 * 256
                # print("col:", col)
                if len(id_renumbering) == 0:
                    self.reverse_id[col] = idx
                else:
                    self.reverse_id[col] = id_renumbering[idx] if idx in id_renumbering else 0
                self.names[idx] = "replicant"
                self.categories[idx] = "replicant"
        else:
            raise Exception("Unknown type of segmentation colors")
    
    def get_seg_mask(self, idPass: np.ndarray, none_id: int = 0, rcnn=None, id_list=None):
        idPass[idPass == None] = none_id
        compressed = idPass.astype(np.uint32)
        compressed = compressed[:, :, 0] + compressed[:, :, 1] * 256 + compressed[:, :, 2] * 256 * 256
        if rcnn is None:
            return np.vectorize(self.reverse_id.get)(compressed).astype(np.int64)
        
        # get bboxes for ret
        all_idx = np.unique(compressed)
        all_idx = all_idx[all_idx != 0]

        real_bboxes = []
        for idx in all_idx:
            mask = compressed == idx
            y, x = np.where(mask)
            bbox = [np.min(x), np.min(y), np.max(x), np.max(y)]
            real_bboxes.append([bbox, idx])
        
        def find_idx(bbox2, rcnn_cat):
            # find bbox with max iou
            max_iou = 0
            max_idx = -1
            for bbox, idx in real_bboxes:
                if rcnn_cat != self.categories[id_list[self.reverse_id[idx]]]:
                    continue
                intersection = max(0.0, min(bbox[2], bbox2[2]) - max(bbox[0], bbox2[0])) * max(0.0, min(bbox[3], bbox2[3]) - max(bbox[1], bbox2[1]))
                union = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - intersection
                iou = intersection / union
                if iou > max_iou:
                    max_iou = iou
                    max_idx = idx
            # print("max_iou:", max_iou, "max_idx:", max_idx)
            return max_idx
        
        ret = np.zeros((idPass.shape[0], idPass.shape[1]), dtype=np.int64)
        labels = rcnn.pred_instances.labels
        masks = rcnn.pred_instances.masks
        scores = rcnn.pred_instances.scores
        bboxes = rcnn.pred_instances.bboxes
        l = len(labels)
        for i in range(l):
            cat = self.global_categories[labels[i]]
            bbox = bboxes[i]
            score = scores[i]
            mask = masks[i]
            # if score < 0.5:
            #     continue
            real_idx = find_idx(bbox, cat)
            if real_idx == -1:
                continue
            ret[mask.cpu().numpy()] = self.reverse_id[real_idx]
        return ret