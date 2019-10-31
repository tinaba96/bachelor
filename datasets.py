import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
import utils.frame_utils as frame_utils

from scipy.misc import imread, imresize

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue
            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)
        print(len(self.image_list))
        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size
        
        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'clean', replicates = replicates)

class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)

class FlyingThingsFlowDisp(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1, validation = False):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs_A   = sorted(glob(join(root, dstype, 'TRAIN/A/*')))
    image_dirs_A_L = sorted([join(f, 'left') for f in image_dirs_A])
    image_dirs_A_R = sorted([join(f, 'right') for f in image_dirs_A])
    image_dirs_B   = sorted(glob(join(root, dstype, 'TRAIN/B/*')))
    image_dirs_B_L = sorted([join(f, 'left') for f in image_dirs_B])
    image_dirs_B_R = sorted([join(f, 'right') for f in image_dirs_B])
    image_dirs_C   = sorted(glob(join(root, dstype, 'TRAIN/C/*')))
    image_dirs_C_L = sorted([join(f, 'left') for f in image_dirs_C])
    image_dirs_C_R = sorted([join(f, 'right') for f in image_dirs_C])
    flow_dirs_A    = sorted(glob(join(root, 'optical_flow/TRAIN/A/*')))
    flow_dirs_A_L  = sorted([join(f, 'into_future/left') for f in flow_dirs_A])
    flow_dirs_B    = sorted(glob(join(root, 'optical_flow/TRAIN/B/*')))
    flow_dirs_B_L  = sorted([join(f, 'into_future/left') for f in flow_dirs_B])
    flow_dirs_C    = sorted(glob(join(root, 'optical_flow/TRAIN/C/*')))
    flow_dirs_C_L  = sorted([join(f, 'into_future/left') for f in flow_dirs_C])
    disp_dirs_A    = sorted(glob(join(root, 'disparity/TRAIN/A/*')))
    disp_dirs_A_L  = sorted([join(f, 'left') for f in disp_dirs_A])
    disp_dirs_B    = sorted(glob(join(root, 'disparity/TRAIN/B/*')))
    disp_dirs_B_L  = sorted([join(f, 'left') for f in disp_dirs_B])
    disp_dirs_C    = sorted(glob(join(root, 'disparity/TRAIN/C/*')))
    disp_dirs_C_L  = sorted([join(f, 'left') for f in disp_dirs_C])

    len_A_L = len(image_dirs_A_L)
    len_B_L = len(image_dirs_B_L)
    len_C_L = len(image_dirs_C_L)
    assert(len_A_L == len(image_dirs_A_R))
    assert(len_B_L == len(image_dirs_B_R))
    assert(len_C_L == len(image_dirs_C_R))
    assert(len_A_L == len(flow_dirs_A_L))
    assert(len_B_L == len(flow_dirs_B_L))
    assert(len_C_L == len(flow_dirs_C_L))
    assert(len_A_L == len(disp_dirs_A_L))
    assert(len_B_L == len(disp_dirs_B_L))
    assert(len_C_L == len(disp_dirs_C_L))

    num_A = int(0.1 * len_A_L)
    num_B = int(0.1 * len_B_L)
    num_C = int(0.1 * len_C_L)

    if (validation):
      image_dirs_L = image_dirs_A_L[:num_A-1] + image_dirs_B_L[:num_B-1] + image_dirs_C_L[:num_C-1]
      image_dirs_R = image_dirs_A_R[:num_A-1] + image_dirs_B_R[:num_B-1] + image_dirs_C_R[:num_C-1]
      flow_dirs = flow_dirs_A_L[:num_A-1] + flow_dirs_B_L[:num_B-1] + flow_dirs_C_L[:num_C-1]
      disp_dirs = disp_dirs_A_L[:num_A-1] + disp_dirs_B_L[:num_B-1] + disp_dirs_C_L[:num_C-1]
    else:
      image_dirs_L = image_dirs_A_L[num_A:] + image_dirs_B_L[num_B:] + image_dirs_C_L[num_C:]
      image_dirs_R = image_dirs_A_R[num_A:] + image_dirs_B_R[num_B:] + image_dirs_C_R[num_C:]
      flow_dirs = flow_dirs_A_L[num_A:] + flow_dirs_B_L[num_B:] + flow_dirs_C_L[num_C:]
      disp_dirs = disp_dirs_A_L[num_A:] + disp_dirs_B_L[num_B:] + disp_dirs_C_L[num_C:]

    self.image_list = []
    self.flow_list = []

    for ldir, rdir, fdir, ddir in zip(image_dirs_L, image_dirs_R, flow_dirs, disp_dirs):
        Limages = sorted( glob(join(ldir, '*.png')) )
        Rimages = sorted( glob(join(rdir, '*.png')) )
        flows = sorted( glob(join(fdir, '*.pfm')) )
        disps = sorted( glob(join(ddir, '*.pfm')) )
        for i in range(len(Limages)-1):
            self.image_list += [ [ Limages[i], Limages[i+1] ] ]
            self.flow_list  += [flows[i]]
            self.image_list += [ [ Limages[i], Rimages[i] ] ]
            self.flow_list  += [disps[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsCleanFlowDispTrain(FlyingThingsFlowDisp):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsCleanFlowDispTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates, validation = False)

class FlyingThingsCleanFlowDispValid(FlyingThingsFlowDisp):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsCleanFlowDispValid, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates, validation = True)

class FlyingChairsFlyingThings(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingchairs', root2 = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.ppm') ) )

    self.flow_list = sorted( glob( join(root, '*.flo') ) )

    assert (len(images)//2 == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = images[2*i]
        im2 = images[2*i + 1]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    image_dirs_A   = sorted(glob(join(root2, dstype, 'TRAIN/A/*')))
    image_dirs_A_L = sorted([join(f, 'left') for f in image_dirs_A])
    image_dirs_A_R = sorted([join(f, 'right') for f in image_dirs_A])
    image_dirs_B   = sorted(glob(join(root2, dstype, 'TRAIN/B/*')))
    image_dirs_B_L = sorted([join(f, 'left') for f in image_dirs_B])
    image_dirs_B_R = sorted([join(f, 'right') for f in image_dirs_B])
    image_dirs_C   = sorted(glob(join(root2, dstype, 'TRAIN/C/*')))
    image_dirs_C_L = sorted([join(f, 'left') for f in image_dirs_C])
    image_dirs_C_R = sorted([join(f, 'right') for f in image_dirs_C])
    disp_dirs_A    = sorted(glob(join(root2, 'disparity/TRAIN/A/*')))
    disp_dirs_A_L  = sorted([join(f, 'left') for f in disp_dirs_A])
    disp_dirs_B    = sorted(glob(join(root2, 'disparity/TRAIN/B/*')))
    disp_dirs_B_L  = sorted([join(f, 'left') for f in disp_dirs_B])
    disp_dirs_C    = sorted(glob(join(root2, 'disparity/TRAIN/C/*')))
    disp_dirs_C_L  = sorted([join(f, 'left') for f in disp_dirs_C])

    len_A_L = len(image_dirs_A_L)
    len_B_L = len(image_dirs_B_L)
    len_C_L = len(image_dirs_C_L)
    assert(len_A_L == len(image_dirs_A_R))
    assert(len_B_L == len(image_dirs_B_R))
    assert(len_C_L == len(image_dirs_C_R))
    assert(len_A_L == len(disp_dirs_A_L))
    assert(len_B_L == len(disp_dirs_B_L))
    assert(len_C_L == len(disp_dirs_C_L))

    num_A = int(0.1 * len_A_L)
    num_B = int(0.1 * len_B_L)
    num_C = int(0.1 * len_C_L)

    image_dirs_L = image_dirs_A_L[num_A:] + image_dirs_B_L[num_B:] + image_dirs_C_L[num_C:]
    image_dirs_R = image_dirs_A_R[num_A:] + image_dirs_B_R[num_B:] + image_dirs_C_R[num_C:]
    disp_dirs = disp_dirs_A_L[num_A:] + disp_dirs_B_L[num_B:] + disp_dirs_C_L[num_C:]

    for ldir, rdir, ddir in zip(image_dirs_L, image_dirs_R, disp_dirs):
        Limages = sorted( glob(join(ldir, '*.png')) )
        Rimages = sorted( glob(join(rdir, '*.png')) )
        disps = sorted( glob(join(ddir, '*.pfm')) )
        for i in range(len(disps)):
            self.image_list += [ [ Limages[i], Rimages[i] ] ]
            self.flow_list += [disps[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingChairsFlyingThingsClean(FlyingChairsFlyingThings):
    def __init__(self, args, is_cropped = False, root = '', root2 = '', replicates = 1):
        super(FlyingChairsFlyingThingsClean, self).__init__(args, is_cropped = is_cropped, root = root, root2 = root2, dstype = 'frames_cleanpass', replicates = replicates)

class FlyingChairs(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/FlyingChairs_release/data', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.ppm') ) )

    self.flow_list = sorted( glob( join(root, '*.flo') ) )

    assert (len(images)//2 == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = images[2*i]
        im2 = images[2*i + 1]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsFlow(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1, validation = False):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs_A   = sorted(glob(join(root, dstype, 'TRAIN/A/*')))
    image_dirs_A_L = sorted([join(f, 'left') for f in image_dirs_A])
    image_dirs_B   = sorted(glob(join(root, dstype, 'TRAIN/B/*')))
    image_dirs_B_L = sorted([join(f, 'left') for f in image_dirs_B])
    image_dirs_C   = sorted(glob(join(root, dstype, 'TRAIN/C/*')))
    image_dirs_C_L = sorted([join(f, 'left') for f in image_dirs_C])
    flow_dirs_A    = sorted(glob(join(root, 'optical_flow/TRAIN/A/*')))
    flow_dirs_A_L  = sorted([join(f, 'into_future/left') for f in flow_dirs_A])
    flow_dirs_B    = sorted(glob(join(root, 'optical_flow/TRAIN/B/*')))
    flow_dirs_B_L  = sorted([join(f, 'into_future/left') for f in flow_dirs_B])
    flow_dirs_C    = sorted(glob(join(root, 'optical_flow/TRAIN/C/*')))
    flow_dirs_C_L  = sorted([join(f, 'into_future/left') for f in flow_dirs_C])

    len_A_L = len(image_dirs_A_L)
    len_B_L = len(image_dirs_B_L)
    len_C_L = len(image_dirs_C_L)
    assert(len_A_L == len(flow_dirs_A_L))
    assert(len_B_L == len(flow_dirs_B_L))
    assert(len_C_L == len(flow_dirs_C_L))

    num_A = int(0.1 * len_A_L)
    num_B = int(0.1 * len_B_L)
    num_C = int(0.1 * len_C_L)

    if (validation):
      image_dirs_L = image_dirs_A_L[:num_A-1] + image_dirs_B_L[:num_B-1] + image_dirs_C_L[:num_C-1]
      flow_dirs = flow_dirs_A_L[:num_A-1] + flow_dirs_B_L[:num_B-1] + flow_dirs_C_L[:num_C-1]
    else:
      image_dirs_L = image_dirs_A_L[num_A:] + image_dirs_B_L[num_B:] + image_dirs_C_L[num_C:]
      flow_dirs = flow_dirs_A_L[num_A:] + flow_dirs_B_L[num_B:] + flow_dirs_C_L[num_C:]

    self.image_list = []
    self.flow_list = []

    for ldir, fdir in zip(image_dirs_L, flow_dirs):
        Limages = sorted( glob(join(ldir, '*.png')) )
        flows = sorted( glob(join(fdir, '*.pfm')) )
        for i in range(len(flows)-1):
            self.image_list += [ [ Limages[i], Limages[i+1] ] ]
            self.flow_list += [flows[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsCleanFlowTrain(FlyingThingsFlow):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsCleanFlowTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates, validation = False)

class FlyingThingsCleanFlowValid(FlyingThingsFlow):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsCleanFlowValid, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates, validation = True)

class FlyingThingsFinalFlowTrain(FlyingThingsFlow):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinalFlowTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates, validation = False)

class FlyingThingsFinalFlowValid(FlyingThingsFlow):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinalFlowValid, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates, validation = True)

class FlyingThingsDisp(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1, validation = False):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs_A   = sorted(glob(join(root, dstype, 'TRAIN/A/*')))
    image_dirs_A_L = sorted([join(f, 'left') for f in image_dirs_A])
    image_dirs_A_R = sorted([join(f, 'right') for f in image_dirs_A])
    image_dirs_B   = sorted(glob(join(root, dstype, 'TRAIN/B/*')))
    image_dirs_B_L = sorted([join(f, 'left') for f in image_dirs_B])
    image_dirs_B_R = sorted([join(f, 'right') for f in image_dirs_B])
    image_dirs_C   = sorted(glob(join(root, dstype, 'TRAIN/C/*')))
    image_dirs_C_L = sorted([join(f, 'left') for f in image_dirs_C])
    image_dirs_C_R = sorted([join(f, 'right') for f in image_dirs_C])
    flow_dirs_A    = sorted(glob(join(root, 'disparity/TRAIN/A/*')))
    flow_dirs_A_L  = sorted([join(f, 'left') for f in flow_dirs_A])
    flow_dirs_B    = sorted(glob(join(root, 'disparity/TRAIN/B/*')))
    flow_dirs_B_L  = sorted([join(f, 'left') for f in flow_dirs_B])
    flow_dirs_C    = sorted(glob(join(root, 'disparity/TRAIN/C/*')))
    flow_dirs_C_L  = sorted([join(f, 'left') for f in flow_dirs_C])

    len_A_L = len(image_dirs_A_L)
    len_B_L = len(image_dirs_B_L)
    len_C_L = len(image_dirs_C_L)
    assert(len_A_L == len(image_dirs_A_R))
    assert(len_B_L == len(image_dirs_B_R))
    assert(len_C_L == len(image_dirs_C_R))
    assert(len_A_L == len(flow_dirs_A_L))
    assert(len_B_L == len(flow_dirs_B_L))
    assert(len_C_L == len(flow_dirs_C_L))

    num_A = int(0.1 * len_A_L)
    num_B = int(0.1 * len_B_L)
    num_C = int(0.1 * len_C_L)

    if (validation):
      image_dirs_L = image_dirs_A_L[:num_A-1] + image_dirs_B_L[:num_B-1] + image_dirs_C_L[:num_C-1]
      image_dirs_R = image_dirs_A_R[:num_A-1] + image_dirs_B_R[:num_B-1] + image_dirs_C_R[:num_C-1]
      flow_dirs = flow_dirs_A_L[:num_A-1] + flow_dirs_B_L[:num_B-1] + flow_dirs_C_L[:num_C-1]
    else:
      image_dirs_L = image_dirs_A_L[num_A:] + image_dirs_B_L[num_B:] + image_dirs_C_L[num_C:]
      image_dirs_R = image_dirs_A_R[num_A:] + image_dirs_B_R[num_B:] + image_dirs_C_R[num_C:]
      flow_dirs = flow_dirs_A_L[num_A:] + flow_dirs_B_L[num_B:] + flow_dirs_C_L[num_C:]

    self.image_list = []
    self.flow_list = []

    for ldir, rdir, fdir in zip(image_dirs_L, image_dirs_R, flow_dirs):
        Limages = sorted( glob(join(ldir, '*.png')) )
        Rimages = sorted( glob(join(rdir, '*.png')) )
        flows = sorted( glob(join(fdir, '*.pfm')) )
        for i in range(len(flows)):
            self.image_list += [ [ Limages[i], Rimages[i] ] ]
            self.flow_list += [flows[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsCleanDispTrain(FlyingThingsDisp):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsCleanDispTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates, validation = False)

class FlyingThingsCleanDispValid(FlyingThingsDisp):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsCleanDispValid, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates, validation = True)

class FlyingThingsFinalDispTrain(FlyingThingsDisp):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinalDispTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates, validation = False)

class FlyingThingsFinalDispValid(FlyingThingsDisp):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinalDispValid, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates, validation = True)

class FlyingThingsCont(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1, validation = False):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs_A   = sorted(glob(join(root, dstype, 'TRAIN/A/*')))
    image_dirs_A_L = sorted([join(f, 'left') for f in image_dirs_A])
    image_dirs_A_R = sorted([join(f, 'right') for f in image_dirs_A])
    image_dirs_B   = sorted(glob(join(root, dstype, 'TRAIN/B/*')))
    image_dirs_B_L = sorted([join(f, 'left') for f in image_dirs_B])
    image_dirs_B_R = sorted([join(f, 'right') for f in image_dirs_B])
    image_dirs_C   = sorted(glob(join(root, dstype, 'TRAIN/C/*')))
    image_dirs_C_L = sorted([join(f, 'left') for f in image_dirs_C])
    image_dirs_C_R = sorted([join(f, 'right') for f in image_dirs_C])
    cont_dirs_A    = sorted(glob(join(root, 'object_contour/TRAIN/A/*')))
    cont_dirs_A_L  = sorted([join(f, 'left') for f in cont_dirs_A])
    cont_dirs_B    = sorted(glob(join(root, 'object_contour/TRAIN/B/*')))
    cont_dirs_B_L  = sorted([join(f, 'left') for f in cont_dirs_B])
    cont_dirs_C    = sorted(glob(join(root, 'object_contour/TRAIN/C/*')))
    cont_dirs_C_L  = sorted([join(f, 'left') for f in cont_dirs_C])

    len_A_L = len(image_dirs_A_L)
    len_B_L = len(image_dirs_B_L)
    len_C_L = len(image_dirs_C_L)
    assert(len_A_L == len(image_dirs_A_R))
    assert(len_B_L == len(image_dirs_B_R))
    assert(len_C_L == len(image_dirs_C_R))
    assert(len_A_L == len(cont_dirs_A_L))
    assert(len_B_L == len(cont_dirs_B_L))
    assert(len_C_L == len(cont_dirs_C_L))

    num_A = int(0.1 * len_A_L)
    num_B = int(0.1 * len_B_L)
    num_C = int(0.1 * len_C_L)

    if (validation):
      image_dirs_L = image_dirs_A_L[:num_A-1] + image_dirs_B_L[:num_B-1] + image_dirs_C_L[:num_C-1]
      image_dirs_R = image_dirs_A_R[:num_A-1] + image_dirs_B_R[:num_B-1] + image_dirs_C_R[:num_C-1]
      cont_dirs = cont_dirs_A_L[:num_A-1] + cont_dirs_B_L[:num_B-1] + cont_dirs_C_L[:num_C-1]
    else:
      image_dirs_L = image_dirs_A_L[num_A:] + image_dirs_B_L[num_B:] + image_dirs_C_L[num_C:]
      image_dirs_R = image_dirs_A_R[num_A:] + image_dirs_B_R[num_B:] + image_dirs_C_R[num_C:]
      cont_dirs = cont_dirs_A_L[num_A:] + cont_dirs_B_L[num_B:] + cont_dirs_C_L[num_C:]

    self.image_list = []
    self.cont_list = []

    for ldir, rdir, cdir in zip(image_dirs_L, image_dirs_R, cont_dirs):
        Limages = sorted( glob(join(ldir, '*.png')) )
        Rimages = sorted( glob(join(rdir, '*.png')) )
        conts = sorted( glob(join(cdir, '*.pgm')) )
        for i in range(len(conts)):
            self.image_list += [ [ Limages[i], Rimages[i] ] ]
            self.cont_list += [conts[i]]

    assert len(self.image_list) == len(self.cont_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    cont = frame_utils.read_gen(self.cont_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    cont = cropper(cont)


    images = np.array(images).transpose(3,0,1,2)
    cont = cont.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    cont = torch.from_numpy(cont.astype(np.float32))

    return [images], [cont]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsCleanContTrain(FlyingThingsCont):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsCleanContTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates, validation = False)

class FlyingThingsCleanContValid(FlyingThingsCont):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsCleanContValid, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates, validation = True)

class FlyingThingsFinalContTrain(FlyingThingsCont):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinalContTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates, validation = False)

class FlyingThingsFinalContValid(FlyingThingsCont):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinalContValid, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates, validation = True)

class ChairsSDHom(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/chairssdhom/data', dstype = 'train', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image1 = sorted( glob( join(root, dstype, 't0/*.png') ) )
    image2 = sorted( glob( join(root, dstype, 't1/*.png') ) )
    self.flow_list = sorted( glob( join(root, dstype, 'flow/*.flo') ) )

    assert (len(image1) == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = image1[i]
        im2 = image2[i]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])
    flow = flow[::-1,:,:]

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class ChairsSDHomTrain(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'train', replicates = replicates)

class ChairsSDHomTest(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTest, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'test', replicates = replicates)

class ImagesFromFolder(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/frames/only/folder', iext = 'png', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.' + iext) ) )
    self.image_list = []
    for i in range(len(images)-1):
        im1 = images[i]
        im2 = images[i+1]
        self.image_list += [ [ im1, im2 ] ]

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    
    images = np.array(images).transpose(3,0,1,2)
    images = torch.from_numpy(images.astype(np.float32))

    return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

  def __len__(self):
    return self.size * self.replicates

'''
import argparse
import sys, os
import importlib
from scipy.misc import imsave
import numpy as np

import datasets
#reload(datasets)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.inference_size = [1080, 1920]
args.crop_size = [384, 512]
args.effective_batch_size = 1

index = 500
v_dataset = datasets.MpiSintelClean(args, True, root='../MPI-Sintel/flow/training')
a, b = v_dataset[index]
im1 = a[0].numpy()[:,0,:,:].transpose(1,2,0)
im2 = a[0].numpy()[:,1,:,:].transpose(1,2,0)
imsave('./img1.png', im1)
imsave('./img2.png', im2)
flow_utils.writeFlow('./flow.flo', b[0].numpy().transpose(1,2,0))

'''
