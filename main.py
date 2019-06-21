import os
import random
import time
import torch
import torchvision
import time
import numpy as np
import warnings
from torch import nn, optim
from model.model import *
from torch.nn import DataParallel
from config import config
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset.dataloader import *
from utils.utils import *
from utils.ProgressBar import *
random.seed(config.seed)
torch.manual_seed(config.seed)
np.random.seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


def main():
    weight_path = config.weights + config.model_name + os.sep + config.description + os.sep + str(config.fold) + os.sep
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    log_path = config.logs + config.model_name + os.sep + config.description + os.sep + str(config.fold) + os.sep
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    submit_path = config.submit + config.model_name + os.sep + config.description + os.sep + str(config.fold) + os.sep
    if not os.path.exists(submit_path):
        os.makedirs(submit_path)
    
    config.write_to_log(log_path + os.sep + 'log.txt')

    #dataset preparing
    train_dataset = customDataset(config.train_data, train = True)
    val_dataset = customDataset(config.test_data, train = True)
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size = config.batch_size * 2, shuffle = False, pin_memory = False)
    #model preparing
    model = get_net(config.num_classes)
    model = DataParallel(model.cuda(), device_ids = config.gpus)
    model.train()
    #optimizer preparing
    optimizer = optim.Adam(model.parameters(), lr = config.lr, amsgrad = True, weight_decay = config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
    #loss preparing
    criterion = nn.CrossEntropyLoss().cuda()

    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf, 0, 0]
    best_precision = 0

    for epoch in range(config.epochs):
        scheduler.step(epoch)
        train_progressor = ProgressBar(log_path, mode="Train",epoch=epoch,total_epoch=config.epochs,model_name=config.model_name,total=len(train_loader))
        for index, (data, label) in enumerate(train_loader):
            train_progressor.current = index
            data = Variable(data).cuda()
            label = Variable(torch.from_numpy(np.asarray(label))).cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            precision1_train,precision2_train = accuracy(output, label ,topk=(1,2))
            train_loss.update(loss.item(),data.size(0))
            train_top1.update(precision1_train[0],data.size(0))
            train_progressor.current_loss = train_loss.avg
            train_progressor.current_top1 = train_top1.avg
            train_progressor()
            #print('train epoch %d iteration %d: loss: %.3f' % (epoch + 1, index + 1, loss.data))
        train_progressor.done()
        val_loss, val_top1 = evaluate(epoch, model, val_loader, criterion, log_path)
        is_best = val_top1 > best_precision
        #print(bool(is_best))
        best_precision = max(val_top1, best_precision)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_name": config.model_name,
                "state_dict": model.state_dict(),
                "best_precision1": best_precision,
                "optimizer": optimizer.state_dict(),
                "fold":config.fold,
                "valid_loss":valid_loss,
            },
            is_best, weight_path, log_path, epoch
        )
        #print('val_loss:')

def evaluate(epoch, model, val_loader, criterion, log_path):
    model.eval()
    val_progressor = ProgressBar(log_path, mode="Val  ",epoch=epoch,total_epoch=config.epochs,model_name=config.model_name,total=len(val_loader))
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for index, (data, label) in enumerate(val_loader):
            val_progressor.current = index
            data = Variable(data).cuda()
            label = Variable(torch.from_numpy(np.asarray(label))).cuda()
            output = model(data)
            loss = criterion(output, label)

            p_top1, p_top2 = accuracy(output, label, topk=(1, 2))
            losses.update(loss.item(), data.size(0))
            top1.update(p_top1[0], data.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
            #print('epoch %d validate iteration %d: loss: %.3f' % (epoch + 1, index + 1, it_loss.data))
            #correct += (output == label).sum()
        val_progressor.done()
    return losses.avg, top1.avg



if __name__ == '__main__':
    main()


