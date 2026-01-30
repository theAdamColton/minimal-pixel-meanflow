import unittest

import torch

from src.supplemental_net import ConvNextV2Loss, BatchedRandomResizedCrop, LPIPSLoss


class TestSuppNet(unittest.TestCase):
    def test_random_resized_crop_batch(self):
        b, c, h, w = 8, 3, 32, 32
        x = torch.randn(b, c, h, w)
        size = 16

        y = BatchedRandomResizedCrop(size)(x)

        self.assertEqual((b, c, size, size), y.shape)

    def test_convnextv2_loss(self):
        loss_fn = ConvNextV2Loss()

        b, c, h, w = 3, 3, 256, 256
        x = torch.randn(b, c, h, w)
        y = torch.randn(b, c, h, w) * 0.01 + x

        loss = loss_fn(x, y)

        self.assertTrue(1, loss.nelement())
        self.assertFalse(loss.isnan())

    def test_lpips_loss(self):
        loss_fn = LPIPSLoss()

        b, c, h, w = 3, 3, 256, 256
        x = torch.randn(b, c, h, w)
        y = torch.randn(b, c, h, w) * 0.01 + x

        loss = loss_fn(x, y)

        self.assertTrue(1, loss.nelement())
        self.assertFalse(loss.isnan())
