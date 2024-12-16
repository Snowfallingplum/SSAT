import os

import torch
from saver import Saver
from options import MakeupOptions
from model import MakeupGAN
from dataset_makeup import MakeupDataset

import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# if use pre-trained model SSAT.pth, please modify line 249 and 256 to dim=2 of softmax function in networks.py
def pair_test():
    # parse options
    parser = MakeupOptions()
    opts = parser.parse()
    opts.phase = 'test_pair'
    opts.dataroot = './examples/images'
    opts.result_dir = './inference_results/'
    opts.checkpoint_dir = './weights'
    
    # data loader
    print('\n--- load dataset ---')
    dataset = MakeupDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(len(train_loader))


    # model
    print('\n--- load model ---')
    model = MakeupGAN(opts)
    #ep0, total_it = model.resume(opts.resume)

    ep0, total_it = model.resume(os.path.join(opts.checkpoint_dir, '00999.pth'), False)
    model.eval()
    print('start pair test')
    # saver for display and output
    saver = Saver(opts)
    for iter, data in enumerate(train_loader):
        with torch.no_grad():
            saver.write_test_pair_img(iter, model,data)

if __name__=='__main__':
    pair_test()
