""" Module contains the GAN wrapper around Generator and Discriminator """

import torch as th
import os
import time
import timeit
import datetime


class GAN:
    """ Unconditional VGAN (at least in this repo)
        args:
            :param gen: Generator of the GAN
            :param dis: Discriminator of the GAN
            :param device: Device to run the model on
            :param alpha: Learning rate for Beta parameter
    """

    def __init__(self, gen, dis, device=th.device("cpu"), alpha=1e-6):
        """ constructor for the class """
        from torch.nn import DataParallel
        from vdb.Gan_networks import Generator, Discriminator

        assert isinstance(gen, Generator), "gen is not an instance of Generator"
        assert isinstance(dis, Discriminator), "dis is not an instance of Discriminator"

        self.gen = gen
        self.dis = dis
        self.device = device
        self.latent_size = self.gen.z_dim

        # Create the Generator and the Discriminator
        if self.device == th.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)

        # state of the object
        self.alpha = alpha

        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()

    def optimize_discriminator(self, dis_optim, noise,
                               real_batch, loss_fn, beta, i_c):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :param beta: current value beta for optimization
        :param i_c: value of information bottleneck
        :return: current loss
        """

        # create a batch of generated samples
        fake_samples = self.gen(noise).detach()

        loss, bottleneck_loss = loss_fn.dis_loss(real_batch, fake_samples, i_c)

        # calculate the total loss:
        total_loss = loss + (beta * bottleneck_loss)

        # optimize discriminator
        dis_optim.zero_grad()
        total_loss.backward()
        dis_optim.step()

        return loss.item(), bottleneck_loss.item()

    def optimize_beta(self, beta, bottleneck_loss):
        """
        perform a step for updating the adaptive beta
        :param beta: old value of beta
        :param bottleneck_loss: current value of bottleneck loss
        :return: beta_new => updated value of beta
        """
        # please refer to the section 4 of the vdb_paper in literature
        # for more information about this.
        # this performs gradient ascent over the beta parameter
        beta_new = max(0, beta + (self.alpha * bottleneck_loss))

        # return the updated beta value:
        return beta_new

    def optimize_generator(self, gen_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # create a batch of generated samples
        fake_samples = self.gen(noise)

        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize generator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()

        return loss.item()

    @staticmethod
    def create_grid(samples, img_file):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for visualization: Tensor
        :param img_file: name of file to write
        :return: None (saves file to the disk)
        """
        from torchvision.utils import save_image
        from numpy import sqrt

        # save the images:
        samples = th.clamp((samples.detach() / 2) + 0.5, min=0, max=1)
        save_image(samples, img_file, nrow=int(sqrt(samples.shape[0])))

    def train(self, data, gen_optim, dis_optim, loss_fn,
              init_beta=0, i_c=0.2, latent_distrib="gaussian",
              start=1, num_epochs=12, feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=64,
              log_dir=None, sample_dir="./samples",
              save_dir="./models"):
        """
        Method for training the network
        :param data: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap this inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap this inside a Scheduler if you want to
        :param loss_fn: Object of GANLoss
        :param init_beta: initial value of beta (required while resuming training)
        :param i_c: value for the information bottleneck in Discriminator
        :param latent_distrib: one of "gaussian" or "uniform"
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note this is absolute and not relative to start
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after these many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: path to directory for saving the loss.log file
        :param sample_dir: path to directory for saving generated samples' grids
        :param save_dir: path to directory for saving the trained models
        :return: None (writes multiple files to disk)
        """
        from vdb.Losses import GANLossWithBottleneck

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()

        assert isinstance(gen_optim, th.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, th.optim.Optimizer), \
            "dis_optim is not an Optimizer"
        assert isinstance(loss_fn, GANLossWithBottleneck), \
            "loss_fn is not an instance of GANLossWithBottleneck"
        assert latent_distrib in ("gaussian", "uniform"), \
            "latent_distrib cannot be interpreted"

        print("Starting the training process ... ")

        # obtain the distribution creator function based on requested
        # latent distribution
        if latent_distrib.lower() == "gaussian":
            distrib = th.randn
        else:
            distrib = th.rand

        # create fixed_input for debugging
        fixed_input = distrib(num_samples,
                              self.latent_size).to(self.device)

        # create a global time counter
        global_time = time.time()

        # initialize beta to provided value
        beta = init_beta  # this gets updated adaptively

        for epoch in range(start, num_epochs + 1):
            start = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            total_batches = len(iter(data))

            limit = int((data_percentage / 100) * total_batches)

            for (i, batch) in enumerate(data, 1):

                # extract current batch of data for training
                images = batch.to(self.device)
                extracted_batch_size = images.shape[0]

                gan_input = distrib(extracted_batch_size,
                                    self.latent_size).to(self.device)

                # optimize the discriminator:
                dis_loss, bottleneck_loss = self.optimize_discriminator(
                    dis_optim, gan_input,
                    images, loss_fn, beta, i_c
                )

                # optimize the beta value:
                beta = self.optimize_beta(beta, bottleneck_loss)

                # optimize the generator:
                gan_input = distrib(extracted_batch_size,  # resample from the latent noise
                                    self.latent_size).to(self.device)
                gen_loss = self.optimize_generator(gen_optim, gan_input,
                                                   images, loss_fn)

                # provide a loss feedback
                if i % int(limit / feedback_factor) == 0 or i == 1:
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] batch: %d  d_loss: "
                          "%f, bottleneck_loss: %f, g_loss: %f, beta: %f"
                          % (elapsed, i, dis_loss, bottleneck_loss, gen_loss, beta))

                    # also write the losses to the log file:
                    if log_dir is not None:
                        log_file = os.path.join(log_dir, "loss.log")
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, "a") as log:
                            log.write(str(dis_loss) + "\t" +
                                      str(bottleneck_loss) + "\t" +
                                      str(gen_loss) + "\n")

                    # create a grid of samples and save it
                    gen_img_file = os.path.join(sample_dir, "gen_" +
                                                str(epoch) + "_" +
                                                str(i) + ".png")

                    # Make sure all the required directories exist
                    # otherwise make them
                    os.makedirs(sample_dir, exist_ok=True)

                    # save the generated samples without calculating gradients
                    dis_optim.zero_grad()
                    gen_optim.zero_grad()
                    with th.no_grad():
                        self.create_grid(self.gen(fixed_input), gen_img_file)

                if i > limit:
                    break

            # calculate the time required for the epoch
            stop = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(save_dir, "GAN_GEN_OPTIM"
                                                   + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(save_dir, "GAN_DIS_OPTIM"
                                                   + str(epoch) + ".pth")

                th.save(self.gen.state_dict(), gen_save_file)
                th.save(self.dis.state_dict(), dis_save_file)
                th.save(gen_optim.state_dict(), gen_optim_save_file)
                th.save(dis_optim.state_dict(), dis_optim_save_file)

                # also save the value of beta in a file:
                # note that this keeps rewriting the value in the file
                with open(os.path.join(save_dir, "beta_value_" + str(epoch) + ".txt"),
                          "w") as beta_file:
                    beta_file.write(str(beta))

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()
