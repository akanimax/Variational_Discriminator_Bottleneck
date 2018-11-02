import torch

from unittest import TestCase
from vdb import Gan_networks as gns


class TestResnetBlock(TestCase):

    def setUp(self):
        # in this block shortcut is learnable
        self.resBlock_1 = gns.ResnetBlock(fin=21, fout=79)

        # in this block shortcut is not learnable
        self.resBlock_2 = gns.ResnetBlock(fin=69, fout=69)

        # print the Resblocks
        print("\nResblock 1:\n%s" % str(self.resBlock_1))
        print("\nResblock 2:\n%s" % str(self.resBlock_2))

    def test_forward(self):
        # test the forward for the first res block
        mock_in = torch.randn(32, 21, 16, 16)
        mock_out = self.resBlock_1(mock_in)
        self.assertEqual(mock_out.shape, (32, 79, 16, 16))
        self.assertEqual(torch.isnan(mock_out).sum().item(), 0)
        self.assertEqual(torch.isinf(mock_out).sum().item(), 0)

        # test the forward for the second res block
        mock_in = torch.randn(32, 69, 16, 16)
        mock_out = self.resBlock_2(mock_in)
        self.assertEqual(mock_out.shape, (32, 69, 16, 16))
        self.assertEqual(torch.isnan(mock_out).sum().item(), 0)
        self.assertEqual(torch.isinf(mock_out).sum().item(), 0)

    def tearDown(self):
        # delete all the computational blocks
        del self.resBlock_1, self.resBlock_2
