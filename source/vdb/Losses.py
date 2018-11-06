""" Module implementing various loss functions """

import torch as th


# =================================================================================
# Loss Interface for the VDB discriminators ...
# =================================================================================

class GANLossWithBottleneck:
    """
        Interface for the VDB loss functions
        Args:
            dis: object of the discriminator used
    """

    def __init__(self, dis):
        from vdb.Gan_networks import Discriminator
        from torch.nn import DataParallel

        # assert that discriminator is an object of dis
        if isinstance(dis, DataParallel):
            assert isinstance(dis.module, Discriminator), \
                "dis is not an instance of Discriminator"
        else:
            assert isinstance(dis, Discriminator), \
                "dis is not an instance of Discriminator"

        # state of the object
        self.dis = dis

    @staticmethod
    def _bottleneck_loss(mus, sigmas, i_c, alpha=1e-8):
        """
        calculate the bottleneck loss for the given mus and sigmas
        :param mus: means of the gaussian distributions
        :param sigmas: stds of the gaussian distributions
        :param i_c: value of bottleneck
        :param alpha: small value for numerical stability
        :return: loss_value: scalar tensor
        """
        # add a small value to sigmas to avoid inf log
        kl_divergence = (0.5 * th.sum((mus ** 2) + (sigmas ** 2)
                                      - th.log((sigmas ** 2) + alpha) - 1, dim=1))

        # calculate the bottleneck loss:
        bottleneck_loss = (th.mean(kl_divergence) - i_c)

        # return the bottleneck_loss:
        return bottleneck_loss

    def dis_loss(self, real_samps, fake_samps, i_c):
        """
        calculate discriminator loss
        :param real_samps: real samples batch
        :param fake_samps: fake samples batch
        :param i_c: value of information bottleneck
        :return: gan_loss, bottleneck_loss: scalar tensors
        """
        raise NotImplementedError("Cannot use this class directly for training")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate generator loss
        :param real_samps: real samples batch
        :param fake_samps: fake samples batch
        :return: gan_loss: scalar tensor
        """
        raise NotImplementedError("Cannot use this class directly for training")

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        raise NotImplementedError("Conditional Interface is not Available yet")

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        raise NotImplementedError("Conditional Interface is not Available yet")


# =================================================================================
# Different types of Losses available ...
# =================================================================================

class StandardGAN(GANLossWithBottleneck):

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        pass

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        pass

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, i_c):
        # calculate the real loss:
        # predictions for real images along with mus and sigmas:
        r_preds, r_mus, r_sigmas = self.dis(real_samps, mean_mode=False)
        real_loss = self.criterion(
            th.squeeze(r_preds),
            th.ones(real_samps.shape[0]).to(fake_samps.device))

        # calculate the fake loss:
        f_preds, f_mus, f_sigmas = self.dis(fake_samps, mean_mode=False)
        fake_loss = self.criterion(
            th.squeeze(f_preds),
            th.zeros(fake_samps.shape[0]).to(real_samps.device))

        # calculate the bottleneck_loss:
        bottle_neck_loss = self._bottleneck_loss(
            th.cat((r_mus, f_mus), dim=0),
            th.cat((r_sigmas, f_sigmas), dim=0), i_c)

        # return final losses
        return (real_loss + fake_loss) / 2, bottle_neck_loss

    def gen_loss(self, _, fake_samps):
        preds, _, _ = self.dis(fake_samps, mean_mode=True)
        return self.criterion(th.squeeze(preds),
                              th.ones(fake_samps.shape[0]).to(fake_samps.device))


class StandardGANWithSigmoid(GANLossWithBottleneck):

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        pass

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        pass

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss
        from torch.nn.functional import sigmoid

        super().__init__(dis)

        # define the criterion object
        self.criterion = BCEWithLogitsLoss()
        self.act = sigmoid

    def dis_loss(self, real_samps, fake_samps, i_c):
        # calculate the real loss:
        # predictions for real images along with mus and sigmas:
        r_preds, r_mus, r_sigmas = self.dis(real_samps, mean_mode=False)
        r_preds = self.act(r_preds)
        real_loss = self.criterion(
            th.squeeze(r_preds),
            th.ones(real_samps.shape[0]).to(real_samps.device))

        # calculate the fake loss:
        f_preds, f_mus, f_sigmas = self.dis(fake_samps, mean_mode=False)
        f_preds = self.act(f_preds)
        fake_loss = self.criterion(
            th.squeeze(f_preds),
            th.zeros(fake_samps.shape[0]).to(fake_samps.device))

        # calculate the bottleneck_loss:
        bottle_neck_loss = self._bottleneck_loss(
            th.cat((r_mus, f_mus), dim=0),
            th.cat((r_sigmas, f_sigmas), dim=0), i_c)

        # return final losses
        return (real_loss + fake_loss) / 2, bottle_neck_loss

    def gen_loss(self, _, fake_samps):
        preds, _, _ = self.dis(fake_samps, mean_mode=True)
        preds = self.act(preds)
        return self.criterion(th.squeeze(preds),
                              th.ones(fake_samps.shape[0]).to(fake_samps.device))


class WGAN_GP(GANLossWithBottleneck):

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        pass

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        pass

    def __init__(self, dis, drift=0.001, use_gp=True):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param reg_lambda: regularisation lambda
        :return: gradient_penalty => scalar tensor
        """
        from torch.autograd import grad

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand(batch_size, 1, 1, 1).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)

        # forward pass
        op, _, _ = self.dis(merged, mean_mode=False)

        # obtain gradient of op wrt. merged
        gradient = grad(outputs=op, inputs=merged, create_graph=True,
                        grad_outputs=th.ones_like(op),
                        retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, i_c, reg_lambda=10):
        # define the (Wasserstein) loss
        fake_out, f_mus, f_sigmas = self.dis(fake_samps, mean_mode=False)
        real_out, r_mus, r_sigmas = self.dis(real_samps, mean_mode=False)

        loss = (th.mean(fake_out) - th.mean(real_out)
                + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            fake_samps.requires_grad = True  # turn on gradients for penalty calculation
            gp = self.__gradient_penalty(real_samps, fake_samps, reg_lambda)
            loss += gp

        # calculate the bottleneck_loss:
        bottleneck_loss = self._bottleneck_loss(
            th.cat((r_mus, f_mus), dim=0),
            th.cat((r_sigmas, f_sigmas), dim=0), i_c)

        # return the losses:
        return loss, bottleneck_loss

    def gen_loss(self, _, fake_samps):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.dis(fake_samps, mean_mode=True)[0])

        return loss


class LSGAN(GANLossWithBottleneck):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, i_c):
        r_preds, r_mus, r_sigmas = self.dis(real_samps, mean_mode=False)
        f_preds, f_mus, f_sigmas = self.dis(fake_samps, mean_mode=False)

        # calculate the loss
        loss = 0.5 * (((th.mean(r_preds) - 1) ** 2)
                      + (th.mean(f_preds)) ** 2)

        # calculate the bottleneck_loss:
        bottleneck_loss = self._bottleneck_loss(
            th.cat((r_mus, f_mus), dim=0),
            th.cat((r_sigmas, f_sigmas), dim=0), i_c)

        return loss, bottleneck_loss

    def gen_loss(self, _, fake_samps):
        return 0.5 * ((th.mean(
            self.dis(fake_samps, mean_mode=True)[0]) - 1) ** 2)

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        pass

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        pass


class HingeGAN(GANLossWithBottleneck):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, i_c):
        r_preds, r_mus, r_sigmas = self.dis(real_samps, mean_mode=False)
        f_preds, f_mus, f_sigmas = self.dis(fake_samps, mean_mode=False)

        loss = (th.mean(th.nn.ReLU()(1 - r_preds)) +
                th.mean(th.nn.ReLU()(1 + f_preds)))

        # calculate the bottleneck_loss:
        bottleneck_loss = self._bottleneck_loss(
            th.cat((r_mus, f_mus), dim=0),
            th.cat((r_sigmas, f_sigmas), dim=0), i_c)

        return loss, bottleneck_loss

    def gen_loss(self, _, fake_samps):
        return -th.mean(self.dis(fake_samps, mean_mode=True)[0])

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        pass

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        pass


class RelativisticAverageHingeGAN(GANLossWithBottleneck):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, i_c):
        # Obtain predictions
        r_preds, r_mus, r_sigmas = self.dis(real_samps, mean_mode=False)
        f_preds, f_mus, f_sigmas = self.dis(fake_samps, mean_mode=False)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

        # calculate the bottleneck_loss:
        bottleneck_loss = self._bottleneck_loss(
            th.cat((r_mus, f_mus), dim=0),
            th.cat((r_sigmas, f_sigmas), dim=0), i_c)

        return loss, bottleneck_loss

    def gen_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds, _, _ = self.dis(real_samps, mean_mode=True)
        f_preds, _, _ = self.dis(fake_samps, mean_mode=True)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        pass

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        pass
