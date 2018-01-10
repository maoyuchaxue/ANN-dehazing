import os
from cgan import CGAN

import tensorflow as tf
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--model_name', type=str, default='default',
                        help='Current model name')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
#    check_folder(args.checkpoint_dir)

    # --result_dir
 #   check_folder(args.result_dir)

    # --result_dir
  #  check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = CGAN(sess,
                epoch=args.epoch,
                batch_size=args.batch_size,
                z_dim=args.z_dim,
                checkpoint_dir=args.checkpoint_dir,
                result_dir=args.result_dir,
                log_dir=args.log_dir,
                model_name=args.model_name)

        # build graph
        gan.build_model()

        # show network architecture
        # show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        # gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
