import torch

from unittest import TestCase
from vdb import Gan_networks as gns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestResnetBlock(TestCase):

    def setUp(self):
        # in this block shortcut is learnable
        self.resBlock_1 = gns.ResnetBlock(fin=21, fout=79).to(device)

        # in this block shortcut is not learnable
        self.resBlock_2 = gns.ResnetBlock(fin=69, fout=69).to(device)

        # print the Resblocks
        print("\nResblock 1:\n%s" % str(self.resBlock_1))
        print("\nResblock 2:\n%s" % str(self.resBlock_2))

    def test_forward(self):
        # test the forward for the first res block
        mock_in = torch.randn(32, 21, 16, 16).to(device)
        mock_out = self.resBlock_1(mock_in)
        self.assertEqual(mock_out.shape, (32, 79, 16, 16))
        self.assertEqual(torch.isnan(mock_out).sum().item(), 0)
        self.assertEqual(torch.isinf(mock_out).sum().item(), 0)

        # test the forward for the second res block
        mock_in = torch.randn(32, 69, 16, 16).to(device)
        mock_out = self.resBlock_2(mock_in)
        self.assertEqual(mock_out.shape, (32, 69, 16, 16))
        self.assertEqual(torch.isnan(mock_out).sum().item(), 0)
        self.assertEqual(torch.isinf(mock_out).sum().item(), 0)

    def tearDown(self):
        # delete all the computational blocks
        del self.resBlock_1, self.resBlock_2


class TestDiscriminator(TestCase):

    def setUp(self):
        # edge case discriminator:
        self.dis_edge = gns.Discriminator(size=4).to(device)

        # normal case discriminator:
        self.dis = gns.Discriminator(size=256,
                                     num_filters=64,
                                     max_filters=512).to(device)

        # print some information:
        print("\nDiscriminator 1:\n%s" % str(self.dis_edge))
        print("\nDiscriminator 2:\n%s" % str(self.dis))

    def test_forward(self):
        # test the edge discriminator:
        mock_in = torch.randn(3, 3, 4, 4).to(device)
        for mean_mode in (True, False):
            mock_out1, mock_out2, mock_out3 = self.dis_edge(mock_in, mean_mode)

            # check the shapes of all the three:
            self.assertEqual(mock_out1.shape, (3, 1))
            self.assertEqual(mock_out2.shape, (3, 32))
            self.assertEqual(mock_out3.shape, (3, 32))
            self.assertGreaterEqual(mock_out3.min().item(), 0)
            self.assertEqual(torch.isnan(mock_out1).sum().item(), 0)
            self.assertEqual(torch.isinf(mock_out1).sum().item(), 0)

        # test the normal discriminator:
        mock_in = torch.randn(16, 3, 256, 256).to(device)
        for mean_mode in (True, False):
            mock_out1, mock_out2, mock_out3 = self.dis(mock_in, mean_mode)

            # check the shapes of all the three:
            self.assertEqual(mock_out1.shape, (16, 1))
            self.assertEqual(mock_out2.shape, (16, 256))
            self.assertEqual(mock_out3.shape, (16, 256))
            self.assertGreaterEqual(mock_out3.min().item(), 0)
            self.assertEqual(torch.isnan(mock_out1).sum().item(), 0)
            self.assertEqual(torch.isinf(mock_out1).sum().item(), 0)

    def tearDown(self):
        # delete all the computational blocks
        del self.dis_edge, self.dis


class TestGenerator(TestCase):

    def setUp(self):
        # edge case generator:
        self.gen_edge = gns.Generator(z_dim=128, size=4).to(device)

        # normal case generator:
        self.gen = gns.Generator(z_dim=8, size=256,
                                 final_channels=64,
                                 max_channels=512).to(device)

        # print some information:
        print("\nGenerator 1:\n%s" % str(self.gen_edge))
        print("\nGenerator 2:\n%s" % str(self.gen))

    def test_forward(self):
        # test the edge discriminator:
        mock_in = torch.randn(3, 128).to(device)

        mock_out = self.gen_edge(mock_in)

        # check the shapes of all the three:
        self.assertEqual(mock_out.shape, (3, 3, 4, 4))
        self.assertEqual(torch.isnan(mock_out).sum().item(), 0)
        self.assertEqual(torch.isinf(mock_out).sum().item(), 0)

        # test the normal discriminator:
        mock_in = torch.randn(16, 8).to(device)

        mock_out = self.gen(mock_in)

        # check the shapes of all the three:
        self.assertEqual(mock_out.shape, (16, 3, 256, 256))
        self.assertEqual(torch.isnan(mock_out).sum().item(), 0)
        self.assertEqual(torch.isinf(mock_out).sum().item(), 0)

    def tearDown(self):
        # delete all the computational blocks
        del self.gen_edge, self.gen
