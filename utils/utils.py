import os
import torch
import shutil 
def save_checkpoint(state, is_best, weight_path, log_path, epoch):
    file_name = weight_path + '_checkpoint.pth.tar'
    torch.save(state, file_name)
    if is_best:
        best_model_path = weight_path + 'model_best.pth.tar'
        #torch.save(state, best_model_path)
        print('Get Better Top1 : %s saving weights to %s' % (state['best_precision1'], best_model_path))
        with open(log_path + 'log.txt', 'a') as f:
            print('Get Better Top1 : %s saving weights to %s' % (state['best_precision1'], best_model_path), file = f)
        shutil.copyfile(file_name, best_model_path)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
