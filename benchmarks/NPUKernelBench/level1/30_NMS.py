import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs Non-Maximum Suppression (NMS) on NPU.
    Pytorch native implemention
    def forward(self, boxes: torch.Tensor, scores: torch.Tensor,
                max_output_size: int, iou_threshold: float,
                scores_threshold: float, pad_to_max_output_size: bool = False):
        boxes_f32 = boxes.float()
        scores_f32 = scores.float()

        score_mask = scores_f32 > scores_threshold
        filtered_boxes = boxes_f32[score_mask]
        filtered_scores = scores_f32[score_mask]
        original_indices = torch.where(score_mask)[0]

        if filtered_boxes.shape[0] == 0:
            num_selected = torch.tensor(0, dtype=torch.int32, device=boxes.device)
            if pad_to_max_output_size:
                selected_indices = torch.zeros(max_output_size, dtype=torch.int32, device=boxes.device)
            else:
                selected_indices = torch.tensor([], dtype=torch.int32, device=boxes.device)
            return selected_indices, num_selected

        sorted_indices = torch.argsort(filtered_scores, descending=True)
        sorted_boxes = filtered_boxes[sorted_indices]
        sorted_original_indices = original_indices[sorted_indices]

        num_boxes = sorted_boxes.shape[0]
        selected_indices_list = []
        suppressed = torch.zeros(num_boxes, dtype=torch.bool, device=boxes.device)

        areas = (sorted_boxes[:, 2] - sorted_boxes[:, 0]) * (sorted_boxes[:, 3] - sorted_boxes[:, 1])

        for i in range(num_boxes):
            if suppressed[i]:
                continue

            selected_indices_list.append(sorted_original_indices[i].item())

            if len(selected_indices_list) >= max_output_size:
                break

            # Vectorized IoU: current box vs all remaining unsuppressed boxes
            rest = torch.arange(i + 1, num_boxes, device=boxes.device)
            if rest.numel() == 0:
                break
            mask = ~suppressed[rest]
            if not mask.any():
                continue
            candidates = rest[mask]

            cur_box = sorted_boxes[i]
            cand_boxes = sorted_boxes[candidates]

            x1_inter = torch.maximum(cur_box[0].expand(cand_boxes.shape[0]), cand_boxes[:, 0])
            y1_inter = torch.maximum(cur_box[1].expand(cand_boxes.shape[0]), cand_boxes[:, 1])
            x2_inter = torch.minimum(cur_box[2].expand(cand_boxes.shape[0]), cand_boxes[:, 2])
            y2_inter = torch.minimum(cur_box[3].expand(cand_boxes.shape[0]), cand_boxes[:, 3])

            inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
            union_area = areas[i] + areas[candidates] - inter_area
            iou = inter_area / union_area.clamp(min=1e-6)

            suppress_mask = iou >= iou_threshold
            suppressed[candidates[suppress_mask]] = True

        num_selected = len(selected_indices_list)

        if pad_to_max_output_size:
            selected_indices = torch.zeros(max_output_size, dtype=torch.int32, device=boxes.device)
            if num_selected > 0:
                selected_indices[:num_selected] = torch.tensor(selected_indices_list, dtype=torch.int32, device=boxes.device)
        else:
            if num_selected > 0:
                selected_indices = torch.tensor(selected_indices_list, dtype=torch.int32, device=boxes.device)
            else:
                selected_indices = torch.tensor([], dtype=torch.int32, device=boxes.device)

        num_selected_tensor = torch.tensor(num_selected, dtype=torch.int32, device=boxes.device)

        return selected_indices, num_selected_tensor

    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        x1_inter = max(box1[0].item(), box2[0].item())
        y1_inter = max(box1[1].item(), box2[1].item())
        x2_inter = min(box1[2].item(), box2[2].item())
        y2_inter = min(box1[3].item(), box2[3].item())

        inter_width = max(0.0, x2_inter - x1_inter)
        inter_height = max(0.0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        iou = inter_area / union_area
        return iou.item()
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, boxes: torch.Tensor, scores: torch.Tensor,
                max_output_size: int, iou_threshold: float,
                scores_threshold: float, pad_to_max_output_size: bool = False):
        """
        Performs Non-Maximum Suppression (NMS) on bounding boxes.

        Args:
            boxes (torch.Tensor): Bounding boxes tensor of shape (N, 4).
            scores (torch.Tensor): Scores tensor of shape (N,).
            max_output_size (int): Maximum number of output boxes.
            iou_threshold (float): IoU threshold for suppression.
            scores_threshold (float): Score threshold to filter boxes.
            pad_to_max_output_size (bool, optional): Pad output to max_output_size.

        Returns:
            tuple: (selected_boxes_indices, num_selected_boxes)
        """
        import torch_npu
        iou_threshold_tensor = torch.tensor(iou_threshold, dtype=torch.float32, device=boxes.device)
        scores_threshold_tensor = torch.tensor(scores_threshold, dtype=torch.float32, device=boxes.device)
        return torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold_tensor,
                                     scores_threshold_tensor, pad_to_max_output_size=pad_to_max_output_size)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "30_NMS.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        boxes_info = inputs[0]
        scores_info = inputs[1]
        max_output_size_info = inputs[2]
        iou_threshold_info = inputs[3]
        scores_threshold_info = inputs[4]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[boxes_info["dtype"]]
        
        boxes = torch.randn(boxes_info["shape"], dtype=dtype)
        scores = torch.randn(scores_info["shape"], dtype=dtype)
        max_output_size = max_output_size_info["value"]
        iou_threshold = iou_threshold_info["value"]
        scores_threshold = scores_threshold_info["value"]
        input_groups.append([boxes, scores, max_output_size, iou_threshold, scores_threshold])
    return input_groups


def get_init_inputs():
    return []
