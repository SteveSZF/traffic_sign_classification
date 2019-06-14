import torch
import os
from config import config
from model import *
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
from dataset.dataloader import customDataset
from torch.nn import DataParallel
def test():
    submit_path = config.submit + config.model_name + os.sep + config.description + os.sep + str(config.fold) + os.sep #save submitted csv results
    weight_path = config.weights + config.model_name + os.sep + config.description + os.sep + str(config.fold) + os.sep 
    csv_map = OrderedDict({'cls':[], 'label':[], 'probability':[]})

    test_loader = DataLoader(customDataset(config.test_data, train = False), batch_size = config.batch_size * 2, shuffle = False, pin_memory = True)

    model = get_net()
    model = DataParallel(model.cuda(), device_ids = config.gpus)
    checkpoint = torch.load(weight_path + 'model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        for index, (data, file_paths) in enumerate(test_loader):
            labels = [int(path.split('/')[-2]) for path in file_paths]
            data = Variable(data).cuda()
            output = model(data)
            smax = nn.Softmax(1)
            smax_out = smax(output)
            _, cls = torch.max(smax_out, 1)
        
            csv_map['cls'].extend(cls)
            csv_map['label'].extend(labels)
            for output in smax_out:
                prob = ";".join([str(i) for i in output.data.tolist()])    
                csv_map['probability'].append(prob)
    result = pd.DataFrame(csv_map)
    result.to_csv(submit_path + 'submit.csv', index = False, header = None)


if __name__ == '__main__':
    test()
