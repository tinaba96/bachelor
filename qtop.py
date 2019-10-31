import torch.nn as nn
import numpy

class qtop(): # quantization operators
    def __init__(self, model, bw):
        self.lev = pow(2., int(bw) - 1)       #quantization levels
        #self.max = (self.lev - 1.) / self.lev #maximum number
        #self.max = (self.lev - 1.) / (self.lev*2.0) #maximum number
        #self.max = (self.lev - 1.) / (self.lev*4.0) #maximum number
        self.max = (self.lev - 1.) / (self.lev*8.0) #maximum number

        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        start_range = 1
        end_range = count_Conv2d
        #end_range = count_Conv2d - 1 #remove last layer
        #end_range = count_Conv2d - 2 #remove 1st layer and last layer
        self.q_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.q_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = 0
        #index = -1 #remove 1st layer
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.q_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def quantization(self):
        self.clampConvParams()
        self.saveParams()
        self.quantizeConvParams()

    def clampConvParams(self):
        for index in range(self.num_of_params):
            #self.target_modules[index].data = \
            #    self.target_modules[index].data.clamp(-1.0, self.max)
            #self.target_modules[index].data = \
            #    self.target_modules[index].data.clamp(-0.5, self.max)
            #self.target_modules[index].data = \
            #    self.target_modules[index].data.clamp(-0.25, self.max)
            self.target_modules[index].data = \
                self.target_modules[index].data.clamp(-0.125, self.max)

    def saveParams(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def quantizeConvParams(self):
        for index in range(self.num_of_params):
            tmp = self.target_modules[index].data
            #tmp = tmp.mul(self.lev).add(0.5).floor().div(self.lev)
            #tmp = tmp.mul(self.lev*2.0).add(0.5).floor().div(self.lev*2.0)
            #tmp = tmp.mul(self.lev*4.0).add(0.5).floor().div(self.lev*4.0)
            tmp = tmp.mul(self.lev*8.0).add(0.5).floor().div(self.lev*8.0)
            self.target_modules[index].data = tmp

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
