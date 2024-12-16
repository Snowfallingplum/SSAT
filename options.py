import argparse
class MakeupOptions():
    def __init__(self):
        self.parser=argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='../makeup_datasets/images/', help='path of data')
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--input_dim', type=int, default=3, help='input_dim')
        self.parser.add_argument('--output_dim', type=int, default=3, help='output_dim')
        self.parser.add_argument('--semantic_dim', type=int, default=18, help='output_dim')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--resize_size', type=int, default=286, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
        self.parser.add_argument('--flip', type=bool, default=True, help='specified if  flipping')
        self.parser.add_argument('--nThreads', type=int, default=4, help='# of threads for data loader')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='makeup', help='folder name to save outputs')
        self.parser.add_argument('--display_dir', type=str, default='./logs/', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='./logs/train_images',
                                 help='path for saving result images and models')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./weights',
                                 help='path for saving result images ')

        self.parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=100, help='freq (epoch) of saving models')


        # weight
        # self.parser.add_argument('--makeup_weight', type=float, default=4, help='makeup_weight')
        self.parser.add_argument('--rec_weight', type=float, default=1, help='rec_weight')
        self.parser.add_argument('--CP_weight', type=float, default=2, help='CP_weight')
        self.parser.add_argument('--GP_weight', type=float, default=1, help='GP_weight')
        self.parser.add_argument('--cycle_weight', type=float, default=1, help='cycle_weight')
        self.parser.add_argument('--adv_weight', type=float, default=1, help='adv_weight')
        self.parser.add_argument('--latent_weight', type=float, default=0, help='latent_weight')
        self.parser.add_argument('--semantic_weight', type=float, default=1, help='semantic_weight')

        # training related
        self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
        self.parser.add_argument('--dis_norm', type=str, default='None',
                                 help='normalization layer in discriminator [None, Instance]')
        self.parser.add_argument('--dis_spectral_norm', type=bool,default=True,
                                 help='use spectral normalization in discriminator')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
        self.parser.add_argument('--n_ep', type=int, default=1000, help='number of epochs')  # 400 * d_iter
        self.parser.add_argument('--n_ep_decay', type=int, default=500,
                                 help='epoch start decay learning rate, set -1 if no decay')  # 200 * d_iter
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--num_residule_block', type=int, default=4, help='num_residule_block')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='lr')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu: e.g. 0 ,use -1 for CPU')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
