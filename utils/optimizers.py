from torch.optim.lr_scheduler import _LRScheduler

class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        return [base_lr * min(step / self.warmup_steps, 1.0) for base_lr in self.base_lrs]
