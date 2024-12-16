import os
import cv2
import numpy as np
import torchvision
from tensorboardX import SummaryWriter


def tensor2img(img):
    img = img.cpu()[0].float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)


def save_imgs(imgs, names, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for img, name in zip(imgs, names):
        img = tensor2img(img)
        img = np.array(img)
        cv2.imwrite(os.path.join(path, name + '.jpg'), img)


class Saver():
    def __init__(self, opts):
        self.display_dir = opts.display_dir
        self.model_dir = opts.checkpoint_dir
        self.result_dir = opts.result_dir


        self.display_freq = opts.display_freq
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq

        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(logdir=self.display_dir)

    # write losses and images to tensorboard
    def write_display(self, total_it, model):
        if (total_it + 1) % self.display_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if
                       not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
            for m in members:
                self.writer.add_scalar(m, getattr(model, m), total_it)

    # save result images
    def write_img(self, ep, model):
        root = os.path.join(self.result_dir,'train')
        if not os.path.exists(root):
            os.makedirs(root)

        if (ep + 1) % self.img_save_freq == 0:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_%05d.jpg' % (root, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.jpg' % (root, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_it, model):
        if (ep + 1) % self.model_save_freq == 0:
            print('--- save the model @ ep %d ---' % (ep))
            model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
        elif ep == -1:
            model.save('%s/last.pth' % self.model_dir, ep, total_it)

    # save test pair images
    def write_test_pair_img(self, iter, model,data):
        root = os.path.join(self.result_dir, 'test_pair')
        if not os.path.exists(root):
            os.makedirs(root)
        test_pair_img = model.test_pair(data)
        img_filename = '%s/gen_%05d.jpg' % (root, iter)
        torchvision.utils.save_image(test_pair_img / 2 + 0.5, img_filename, nrow=1)
