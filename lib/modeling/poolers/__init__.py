import torch
import torch.nn as nn

from torchvision.ops import roi_align

class Pooler(nn.Module):
    def __init__(self, output_size, scales, sampling_ratio, canonical_level=4):
        super(Pooler, self).__init__()
        self.output_size = output_size
        self.scales = scales
        self.sampling_ratio = sampling_ratio
        # The following line is commented as it is not used in this simplified version
        # self.map_levels = LevelMapper(lvl_min, lvl_max, canonical_level=canonical_level)

    def convert_to_roi_format(self, boxes):
        concat_boxes = torch.cat([b.bbox for b in boxes], dim=0) if isinstance(boxes, list) else torch.cat([b for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat([torch.full((len(b), 1), i, dtype=dtype, device=device) for i, b in enumerate(boxes)], dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        num_levels = len(self.scales)
        rois = self.convert_to_roi_format(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros((num_rois, num_channels, output_size, output_size), dtype=dtype, device=device)
        no_grad_level = []

        # Iterate over each level
        for level, (per_level_feature, scale) in enumerate(zip(x, self.scales)):
            spatial_scale = scale
            roi_aligned = roi_align(per_level_feature, rois, self.output_size, spatial_scale, self.sampling_ratio)
            level_mask = torch.full((roi_aligned.size(0),), level, dtype=torch.int64, device=device)
            result = result + roi_aligned
            if roi_aligned.numel() == 0:
                no_grad_level.append(level)

        return result, no_grad_level


