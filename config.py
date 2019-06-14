
class Config(object):
    def __init__(self):
        #1.string parameters
        #train_data = "/raid/traffic_sign/GTSRB_crop/train/"
        #test_data = "/raid/traffic_sign/GTSRB_crop/test/"
        #val_data = "/raid/traffic_sign/GTSRB_crop/test/"
        self.train_data = "/dataset/speed_limitation/train/"
        self.test_data = "/dataset/speed_limitation/test/"
        self.val_data = "/dataset/speed_limitation/test/"
        self.model_name = "vgg11"
        self.fold = 0 
        self.weights = "./checkpoints/"
        #best_models = weights + "best_model/"
        self.submit = "./submit/"
        self.logs = "./logs/"
        self.gpus = [0]
        self.description = "speed_limitation" 

        #2.numeric parameters
        self.epochs = 10
        self.batch_size = 16
        self.img_height = 64
        self.img_width = 64
        self.num_classes = 9
        self.seed = 888
        self.lr = 1e-4
        self.lr_decay = 1e-4
        self.weight_decay = 1e-4
    
    def write_to_log(self, log_file):
        info = "train_data = %s\ntest_data = %s\nval_data = %s\nmodel_name = %s\nfold = %d\ngpus = %s\ndescription = %s\nepochs = %d\nbatch_size = %d\nimg_height = %d\nimg_widght = %d\nnum_classes = %d\nlr = %f\nlr_decay = %f\nweight_decay = %f\n" \
                    % (config.train_data, config.test_data, config.val_data, config.model_name, config.fold, str(config.gpus), config.description, \
                    config.epochs, config.batch_size, config.img_height, config.img_width, config.num_classes, config.lr, config.lr_decay, config.weight_decay)
        with open(log_file, 'w') as f:
            print(info, file = f)

config = Config()


if __name__ == "__main__":
    config.write_to_log('/home/nio/log.txt')