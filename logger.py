import os
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, opt):
        os.makedirs(opt.log_dir, exist_ok=True)
        self.writer = SummaryWriter(opt.log_dir)
    
    def scalarSummary(self, avg_loss, acc, epoch, phase):
        self.writer.add_scalar(f'{phase}_avg_loss', avg_loss, epoch)
        self.writer.add_scalar(f'{phase}_accuracy', acc, epoch)
        self.writer.flush()
