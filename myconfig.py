import os.path

class Myconfig(object):
    def __init__(self):
        self.batch_num    = 500
        self.root         = "/home/dhzeng/AVIDEO/"
        self.input        = 10
        self.epoch        = 10
        self.path         = os.path.join(self.root, "Embedding/ccca_vgg_inception"+str(self.input)+"_02/")
        print(self.path)
        self.feat_root    = os.path.join(self.root, "")
        self.fold_root    = os.path.join(self.root, 'Data')
        self.lab_root     = os.path.join(self.root, "Data/VEGAS_classes/")
        self.loademb_root = os.path.join(self.root, "Embedding/")
        self.result_root  = os.path.join(self.root, "Result/")
        self.fold_num     = 5
        self.audio_input  = 128
        self.rgb_input    = 1024 
        self.margin_num   = 0.4
