from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
VENDOR = ROOT / "external/GF-Screen/docker_submission/vendor"


def load_vendor():
    sys.path.insert(0, str(VENDOR))
    for name in tuple(sys.modules):
        if name == "detectron2" or name.startswith("detectron2."):
            sys.modules.pop(name)


class VendorDetectron2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(f"torch unavailable: {error}")
        cls.torch = torch

    def setUp(self):
        load_vendor()

    def test_surface_is_importable_without_full_detectron2(self):
        from detectron2.layers import Conv2d, DeformConv, ShapeSpec, get_norm
        from detectron2.modeling import BACKBONE_REGISTRY, Backbone
        from detectron2.utils.file_io import PathManager

        self.assertTrue(Conv2d.__module__.startswith("detectron2"))
        self.assertTrue(DeformConv.__module__.startswith("detectron2"))
        self.assertEqual(ShapeSpec(channels=4, stride=2).channels, 4)
        self.assertEqual(get_norm("GN", 32).num_groups, 32)
        self.assertEqual(BACKBONE_REGISTRY.name, "BACKBONE")
        self.assertTrue(issubclass(Backbone, object))
        self.assertEqual(PathManager.get_local_path("checkpoint.pt"), "checkpoint.pt")

    def test_conv_matches_detectron2_inference_order(self):
        from detectron2.layers import Conv2d

        torch = self.torch
        norm = torch.nn.GroupNorm(1, 2)
        layer = Conv2d(1, 2, 1, norm=norm, activation=torch.relu)
        value = torch.randn(1, 1, 3, 3)
        expected = torch.relu(
            norm(torch.nn.functional.conv2d(value, layer.weight, layer.bias))
        )
        self.assertTrue(torch.equal(layer(value), expected))

    def test_deform_conv_fails_closed_if_graph_changes(self):
        from detectron2.layers import DeformConv

        torch = self.torch
        layer = DeformConv(1, 1, 3)
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            layer(torch.zeros(1, 1, 5, 5), torch.zeros(1, 18, 3, 3))

    def test_timm_surface_is_inference_compatible(self):
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_

        torch = self.torch
        self.assertEqual(to_2tuple(3), (3, 3))
        value = torch.ones(2, 3)
        self.assertTrue(torch.equal(DropPath(0.5).eval()(value), value))
        target = torch.empty(16)
        self.assertIs(trunc_normal_(target, std=0.02), target)

    def test_fvcore_weight_initializers_match_required_contract(self):
        from fvcore.nn import weight_init

        torch = self.torch
        layer = torch.nn.Conv2d(1, 2, 3)
        weight_init.c2_xavier_fill(layer)
        self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))
        weight_init.c2_msra_fill(layer)
        self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))


if __name__ == "__main__":
    unittest.main()
