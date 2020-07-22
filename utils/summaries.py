import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
# from dataloaders.utils import decode_seg_map_sequence
from DR_deeplab_xception.dataloaders.utils import decode_seg_map_sequence
class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(torch.max(output[:3], 1)[1].detach().cpu().numpy(), 3, normalize=False,range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(torch.squeeze(target[:3], 1).detach().cpu().numpy(), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)