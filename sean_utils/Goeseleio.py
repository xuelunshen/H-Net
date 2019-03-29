import numpy as np
from PIL import Image

class Goeseleio:
    def __init__(self, batch_size, datafile_path, datanum):
        """
        data io for brown benchmark
        :param batch_size: batch size
        :param datafile_path: file path
        :param datanum: data number
        """
        datalist_path = 'm50_' + datanum + '_' + datanum + '_0.txt' # like m50_500000_500000_0.txt
        self.batch_size = batch_size
        self.datafile_path = datafile_path # like C:\DL\data\DOG\liberty\
        self.datalist_path = datafile_path + datalist_path # like C:\DL\data\DOG\liberty\m50_500000_500000_0.txt
        self.datalist = open(self.datalist_path, 'r').readlines()
        self.datanum = len(self.datalist)
        self.batchnum = self.datanum // self.batch_size
        self.bmprow = 16
        self.bmpcolumn = 16
        self.bmppatchnum = self.bmprow * self.bmpcolumn # means 16*16
        self.patchwidth = 64
        self.patchheight = 64
        self.patchchannel = 1
        self.idx = 0
        self.batchphoto1 = np.zeros((self.batch_size, self.patchwidth, self.patchheight, self.patchchannel))
        self.batchphoto2 = np.zeros((self.batch_size, self.patchwidth, self.patchheight, self.patchchannel))
        self.batchlabel = np.zeros(self.batch_size, dtype = np.int8)
        self.batchphotoid1 = np.zeros(self.batch_size, dtype = np.int8)
        self.batchphotoid2 = np.zeros(self.batch_size, dtype = np.int8)
    
    def getBatchData(self):
        """
        get image matrix in batch size
        :return: data
        """
        for i in range(self.batch_size):
            p1, p2, label, pid1, pid2 = self.getPatchPair()
            self.batchphoto1[i, :, :, :] = p1
            self.batchphoto2[i, :, :, :] = p2
            self.batchlabel[i] = label
            self.batchphotoid1[i] = pid1
            self.batchphotoid2[i] = pid2
        return self.batchphoto1, self.batchphoto2, self.batchlabel, self.batchphotoid1, self.batchphotoid2
    
    def getPatchPair(self):
        """
        get a pair patch by idx
        the idx will increment each a data pick up
        :return: data in pair
        """
        # get a patch pair by the self.idx
        idx = self.idx
        # none sense code, i only want to print it
        if idx == (self.datanum - 1):
            print('this is the final data set idx:{:d}'.format(idx))
            
        if idx == self.datanum:
            print('use out of data so reset the idx:{:d} to 0'.format(idx))
            idx = 0
        elif idx > self.datanum:
            print('idx {:d} should not larger than datanum {:d}'.format(idx, self.datanum))
            exit(-1)
        p1, p2, label, pid1, pid2 = self.getPatchPairbyidx(idx)
        self.idx = idx + 1
        return p1, p2, label, pid1, pid2
    
    def getPatchPairbyidx(self, idx):
        """
        get a pair patch by a certain idx
        :param idx: index
        :return: a pair data in idx
        """
        # if we have a idx, we could get a line like
        # 358580 128293 0 442312 158432 0 0
        # patchID1 3DpointID1 unused1 patchID2 3DpointID2 unused2
        line = self.datalist[idx].strip().split(' ')
        pid1 = int(line[0])
        pid2 = int(line[3])
        if line[1] == line[4]: label = 1
        else: label = -1
        # else: label = 0
        # if we get pid, then we could get correspond image patch
        p1 = self.getPatchbypid(int(pid1))
        p2 = self.getPatchbypid(int(pid2))
        return p1, p2, label, pid1, pid2
    
    def getPatchbypid(self, pid):
        """
        get a pair by photo id
        :param pid: photo id
        :return: a patch data in certain photo id
        """
        raw_id = pid // self.bmppatchnum

        # get likes 'patches0000.bmp'
        bmp_id = ("%04d" % raw_id)
        bmp_name = 'patches' + bmp_id + '.bmp'
        bmp_path = self.datafile_path + bmp_name
        bmp_img = Image.open(bmp_path)

        idx = pid - raw_id * self.bmppatchnum
        y = idx // self.bmprow
        x = idx - y * self.bmprow

        # The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
        bbox = [x * self.patchwidth, y *self.patchheight, x *self.patchwidth + self.patchwidth, y *self.patchheight + self.patchheight]
        patch = bmp_img.crop(bbox)
        patch = self.ImageToMatrix(patch, self.patchwidth, self.patchheight, self.patchchannel)
        return patch
    
    @staticmethod
    def ImageToMatrix(im, width, height, channel):
        """
        change PIL.Image into matrix
        :param im: PIL.Image
        :param width: width
        :param height: height
        :param channel: Channel
        :return: a image matrix
        """
        # im = im.resize((width, height))
        # im = list(im.getdata())
        _im = np.reshape(im, (width, height, channel))
        _im = _im * 1.0 / 255.0
        return _im

        