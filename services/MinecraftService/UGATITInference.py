import sys
sys.path.append("UGATIT")
from UGATIT import UGATIT
import argparse
from utils import *

from glob import glob
import imageio
import io
import cv2

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='UGATIT/results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def getUGATITTransform(input_image, dataset):
    image = imageio.imread(io.BytesIO(input_image))
    [h,w,c] = np.shape(image)
    scale = 1
    if((h > 512) or (w > 512)):
        if (h > w):
            scale = h // 512 + 1
        else:
            scale = w // 512 + 1
    image = cv2.resize(image, dsize=(int(w/scale), int(h/scale)))
    t_image = torch.from_numpy(np.expand_dims(np.transpose(image, (2, 0, 1)), 0)).cuda().float()
    args = parse_args()
    args.light = True
    args.dataset=dataset
    gan = UGATIT(args)
    gan.build_model()
    model_list = glob(os.path.join(gan.result_dir, gan.dataset, 'model', '*.pt'))
    if not len(model_list) == 0:
        model_list.sort()
        iter = int(model_list[-1].split('_')[-1].split('.')[0])
        gan.load(os.path.join(gan.result_dir, gan.dataset, 'model'), iter)
        print(" [*] Load SUCCESS")
    else:
        print(" [*] Load FAILURE")
        return
    gan.genA2B.eval(), gan.genB2A.eval()
    real_A = t_image
    fake_A2B, _, fake_A2B_heatmap = gan.genA2B(real_A)
    result = np.transpose(np.squeeze(fake_A2B.data.cpu().numpy()), (1,2,0))
    return result


if __name__ == '__main__':
    with open('../1.jpg', 'rb') as f:
        getUGATITTransform(f.read(), "minecraft_landscape")
