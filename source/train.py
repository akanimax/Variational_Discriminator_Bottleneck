""" script for training the VDB-GAN on given dataset """

import argparse

import torch as th
from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# enable fast training
cudnn.benchmark = True

# set seed = 3
th.manual_seed(seed=3)


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        default=None,
                        help="pretrained weights file for generator")

    parser.add_argument("--discriminator_file", action="store", type=str,
                        default=None,
                        help="pretrained_weights file for discriminator")

    parser.add_argument("--images_dir", action="store", type=str,
                        default="../data/celeba",
                        help="path for the images directory")

    parser.add_argument("--folder_distributed_dataset", action="store", type=bool,
                        default=False, help="path for the images directory")

    parser.add_argument("--sample_dir", action="store", type=str,
                        default="samples/1/",
                        help="path for the generated samples directory")

    parser.add_argument("--model_dir", action="store", type=str,
                        default="models/1/",
                        help="path for saved models directory")

    parser.add_argument("--loss_function", action="store", type=str,
                        default="wgan-gp",
                        help="loss function to be used: " +
                             "'hinge', 'relativistic-hinge'," +
                             " 'standard-gan', 'standard-gan_with-sigmoid'," +
                             " 'wgan-gp', 'lsgan'")

    parser.add_argument("--size", action="store", type=int,
                        default=128,
                        help="Size of the generated image " +
                             "(must be a power of 2 and >= 4)")

    parser.add_argument("--latent_distrib", action="store", type=str,
                        default="gaussian",
                        help="Type of latent distribution " +
                             "to be used 'uniform' or 'gaussian'")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=512,
                        help="latent size for the generator")

    parser.add_argument("--final_channels", action="store", type=int,
                        default=64,
                        help="starting number of channels in the networks")

    parser.add_argument("--max_channels", action="store", type=int,
                        default=1024,
                        help="maximum number of channels in the network")

    parser.add_argument("--init_beta", action="store", type=float,
                        default=0,
                        help="initial value of beta")

    parser.add_argument("--i_c", action="store", type=float,
                        default=0.2,
                        help="value of information bottleneck")

    parser.add_argument("--batch_size", action="store", type=int,
                        default=32,
                        help="batch_size for training")

    parser.add_argument("--start", action="store", type=int,
                        default=1,
                        help="starting epoch number")

    parser.add_argument("--num_epochs", action="store", type=int,
                        default=3,
                        help="number of epochs for training")

    parser.add_argument("--feedback_factor", action="store", type=int,
                        default=200,
                        help="number of logs to generate per epoch")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=64,
                        help="number of samples to generate for creating the grid" +
                             " should be a square number preferably")

    parser.add_argument("--checkpoint_factor", action="store", type=int,
                        default=1,
                        help="save model per n epochs")

    parser.add_argument("--g_lr", action="store", type=float,
                        default=0.0001,
                        help="learning rate for generator")

    parser.add_argument("--d_lr", action="store", type=float,
                        default=0.0001,
                        help="learning rate for discriminator")

    parser.add_argument("--data_percentage", action="store", type=float,
                        default=100,
                        help="percentage of data to use")

    parser.add_argument("--num_workers", action="store", type=int,
                        default=3,
                        help="number of parallel workers for reading files")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    from vdb.Gan import GAN
    from vdb.Gan_networks import Generator, Discriminator
    from data_processing.DataLoader import FlatDirectoryImageDataset, \
        get_transform, get_data_loader, FoldersDistributedDataset
    from vdb.Losses import WGAN_GP, HingeGAN, RelativisticAverageHingeGAN, \
        StandardGAN, LSGAN, StandardGANWithSigmoid

    # create a data source:
    if args.folder_distributed_dataset:
        data_extractor = FoldersDistributedDataset
    else:
        data_extractor = FlatDirectoryImageDataset

    dataset = data_extractor(args.images_dir,
                             get_transform((args.size, args.size)))

    print("Total number of images in the dataset:", len(dataset))

    data = get_data_loader(dataset, args.batch_size, args.num_workers)

    # create the Generator and Discriminator objects:
    generator = Generator(args.latent_size, args.size,
                          args.final_channels, args.max_channels).to(device)

    discriminator = Discriminator(args.size, args.final_channels,
                                  args.max_channels).to(device)

    # create a gan from these
    vdb_gan = GAN(
        gen=generator,
        dis=discriminator,
        device=device)

    if args.generator_file is not None:
        # load the weights into generator
        print("loading generator weights from:", args.generator_file)
        vdb_gan.gen.load_state_dict(th.load(args.generator_file))

    print("Generator Configuration: ")
    print(vdb_gan.gen)

    if args.discriminator_file is not None:
        # load the weights into discriminator
        print("loading discriminator weights from:", args.discriminator_file)
        vdb_gan.dis.load_state_dict(th.load(args.discriminator_file))

    print("Discriminator Configuration: ")
    print(vdb_gan.dis)

    # create optimizer for generator:
    gen_optim = th.optim.RMSprop(vdb_gan.gen.parameters(), lr=args.g_lr)

    dis_optim = th.optim.RMSprop(vdb_gan.dis.parameters(), lr=args.d_lr)

    loss_name = args.loss_function.lower()

    if loss_name == "hinge":
        loss = HingeGAN
    elif loss_name == "relativistic-hinge":
        loss = RelativisticAverageHingeGAN
    elif loss_name == "standard-gan":
        loss = StandardGAN
    elif loss_name == "standard-gan_with-sigmoid":
        loss = StandardGANWithSigmoid
    elif loss_name == "wgan-gp":
        loss = WGAN_GP
    elif loss_name == "lsgan":
        loss = LSGAN
    else:
        raise Exception("Unknown loss function requested")

    # train the GAN
    vdb_gan.train(
        data=data,
        gen_optim=gen_optim,
        dis_optim=dis_optim,
        loss_fn=loss(vdb_gan.dis),
        init_beta=args.init_beta,
        i_c=args.i_c,
        latent_distrib=args.latent_distrib,
        start=args.start,
        num_epochs=args.num_epochs,
        feedback_factor=args.feedback_factor,
        checkpoint_factor=args.checkpoint_factor,
        data_percentage=args.data_percentage,
        num_samples=args.num_samples,
        log_dir=args.model_dir,
        sample_dir=args.sample_dir,
        save_dir=args.model_dir
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
