import random

from unittest import TestCase
from data_processing import DataLoader as DL


class TestImageDataset(TestCase):

    def setUp(self):
        self.dataset = DL.FlatDirectoryImageDataset(
            data_dir="../data/plants",
            transform=DL.get_transform((128, 128))
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), 3826)

    def test_item(self):
        # pluck a random item from this list:
        random_element = self.dataset[random.randint(0, len(self.dataset))]

        self.assertEqual(random_element.shape, (3, 128, 128))
        self.assertGreaterEqual(random_element.min(), -1)
        self.assertLessEqual(random_element.max(), 1)

    def test_image(self):
        # pluck a random item from this list:
        random_element = self.dataset[random.randint(0, len(self.dataset))]

        # we display the image here:
        import matplotlib.pyplot as plt
        plt.imshow(random_element.permute(1, 2, 0) / 2 + 0.5)
        plt.show()

    def tearDown(self):
        # delete the self->dataset handle
        del self.dataset


class TestImageDataloader(TestCase):

    def setUp(self):
        self.dataloader = DL.get_data_loader(
            DL.FlatDirectoryImageDataset(
                data_dir="../data/plants",
                transform=DL.get_transform((128, 128))
            ),
            batch_size=32,
            num_workers=3
        )

    def test_data_batch(self):
        # obtain a batch of data:
        test_batch = iter(self.dataloader).next()
        self.assertEqual(test_batch.shape, (32, 3, 128, 128))
        self.assertGreaterEqual(test_batch.min(), -1)
        self.assertLessEqual(test_batch.max(), 1)
