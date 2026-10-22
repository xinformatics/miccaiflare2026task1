import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))
from materialize_stunet_gf_ensemble import materialize, parse_configuration


class MaterializeSTUNetGFEnsembleTest(unittest.TestCase):
    def test_configuration_parser_is_closed(self):
        self.assertEqual(parse_configuration("stunet_standalone"), (False, None, None, None, None))
        self.assertEqual(parse_configuration("stunet_mm1000"), (False, 1000, None, None, None))
        self.assertEqual(parse_configuration("stunet_ftu_s10_mm3000"), (False, 3000, 0.10, None, None))
        self.assertEqual(parse_configuration("gf_union_stu_mm1000"), (True, 1000, None, None, None))
        self.assertEqual(parse_configuration("gf_union_stu_ftu_s10_mm3000"), (True, 3000, 0.10, None, None))
        self.assertEqual(parse_configuration("stunet_p40_mm1000"), (False, 1000, None, 0.40, None))
        self.assertEqual(parse_configuration("gf_union_stu_p60_mm3000"), (True, 3000, None, 0.60, None))
        self.assertEqual(parse_configuration("gf_union_stu_p30_ftu_s10_mm3000"), (True, 3000, 0.10, 0.30, None))
        self.assertEqual(parse_configuration("gf_anchor_confirm_either_any"), (False, None, None, None, "either"))
        self.assertEqual(parse_configuration("gf_anchor_confirm_either_p70"), (False, None, None, 0.70, "either"))
        with self.assertRaisesRegex(ValueError, "invalid"):
            parse_configuration("gf_union_stu_ftu_s50_mm0")

    def test_materializer_replays_supported_whole_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dirs={name:root/name for name in ("anchor","stunet","freetumor","output")}
            for directory in dirs.values(): directory.mkdir()
            name="case.nii.gz";affine=np.diag([10.,10.,10.,1.]);shape=(6,6,6)
            anchor=np.zeros(shape,dtype=np.uint8);anchor[0,0,0]=1
            stunet=np.zeros(shape,dtype=np.uint8);stunet[2:4,2:4,2:4]=1
            support=np.zeros(shape,dtype=np.uint8);support[2,2,2]=1
            for role,array in (("anchor",anchor),("stunet",stunet),("freetumor",support)):
                nib.save(nib.Nifti1Image(array,affine),dirs[role]/name)
            materialize(dirs["anchor"],dirs["stunet"],dirs["freetumor"],
                        "gf_union_stu_ftu_s10_mm500",dirs["output"],1)
            result=np.asanyarray(nib.load(str(dirs["output"]/name)).dataobj)
        self.assertTrue(result[0,0,0])
        self.assertTrue(result[2:4,2:4,2:4].all())

    def test_materializer_replays_dual_confirmed_anchor_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dirs={name:root/name for name in ("anchor","stunet","freetumor","output")}
            for directory in dirs.values(): directory.mkdir()
            name="case.nii.gz";affine=np.eye(4);shape=(6,6,6)
            anchor=np.zeros(shape,dtype=np.uint8);anchor[1:3,1:3,1:3]=1;anchor[4:6,4:6,4:6]=1
            stunet=np.zeros(shape,dtype=np.uint8);stunet[1,1,1]=1
            support=np.zeros(shape,dtype=np.uint8);support[4,4,4]=1
            for role,array in (("anchor",anchor),("stunet",stunet),("freetumor",support)):
                nib.save(nib.Nifti1Image(array,affine),dirs[role]/name)
            materialize(dirs["anchor"],dirs["stunet"],dirs["freetumor"],
                        "gf_anchor_confirm_either_any",dirs["output"],1)
            result=np.asanyarray(nib.load(str(dirs["output"]/name)).dataobj)
        self.assertTrue(result[1:3,1:3,1:3].all())
        self.assertTrue(result[4:6,4:6,4:6].all())

    def test_materializer_replays_probability_confirmed_anchor_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dirs={name:root/name for name in ("anchor","stunet","freetumor","output")}
            for directory in dirs.values(): directory.mkdir()
            name="case.nii.gz";affine=np.eye(4);shape=(6,6,6)
            anchor=np.zeros(shape,dtype=np.uint8);anchor[1:3,1:3,1:3]=1;anchor[4:6,4:6,4:6]=1
            stunet=np.zeros(shape,dtype=np.uint8)
            support=np.zeros(shape,dtype=np.uint8);support[4,4,4]=1
            for role,array in (("anchor",anchor),("stunet",stunet),("freetumor",support)):
                nib.save(nib.Nifti1Image(array,affine),dirs[role]/name)
            probability=np.zeros(shape,dtype=np.float32);probability[1,1,1]=0.8
            np.savez_compressed(
                dirs["stunet"] / "case.prob16.npz",
                lesion_probability_uint16=np.rint(probability*65535).astype(np.uint16),
            )
            materialize(dirs["anchor"],dirs["stunet"],dirs["freetumor"],
                        "gf_anchor_confirm_either_p70",dirs["output"],1)
            result=np.asanyarray(nib.load(str(dirs["output"]/name)).dataobj)
        self.assertTrue(result[1:3,1:3,1:3].all())
        self.assertTrue(result[4:6,4:6,4:6].all())

    def test_materializer_replays_quantized_probability_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dirs={name:root/name for name in ("anchor","stunet","output")}
            for directory in dirs.values(): directory.mkdir()
            name="case.nii.gz";affine=np.diag([10.,10.,10.,1.]);shape=(6,6,6)
            anchor=np.zeros(shape,dtype=np.uint8)
            stunet=np.zeros(shape,dtype=np.uint8)
            probability=np.zeros(shape,dtype=np.float32);probability[2:4,2:4,2:4]=0.65
            for role,array in (("anchor",anchor),("stunet",stunet)):
                nib.save(nib.Nifti1Image(array,affine),dirs[role]/name)
            quantized=np.rint(probability*65535).astype(np.uint16)
            np.savez_compressed(
                dirs["stunet"]/"case.prob16.npz",
                lesion_probability_uint16=quantized,
            )
            materialize(dirs["anchor"],dirs["stunet"],None,
                        "stunet_p60_mm1000",dirs["output"],1)
            result=np.asanyarray(nib.load(str(dirs["output"]/name)).dataobj)
        self.assertTrue(result[2:4,2:4,2:4].all())

    def test_materializer_can_replay_standalone_without_anchor_union(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dirs={name:root/name for name in ("anchor","stunet","output")}
            for directory in dirs.values(): directory.mkdir()
            name="case.nii.gz";affine=np.diag([10.,10.,10.,1.]);shape=(6,6,6)
            anchor=np.zeros(shape,dtype=np.uint8);anchor[0,0,0]=1
            stunet=np.zeros(shape,dtype=np.uint8);stunet[2:4,2:4,2:4]=1
            for role,array in (("anchor",anchor),("stunet",stunet)):
                nib.save(nib.Nifti1Image(array,affine),dirs[role]/name)
            materialize(dirs["anchor"],dirs["stunet"],None,
                        "stunet_standalone",dirs["output"],1)
            result=np.asanyarray(nib.load(str(dirs["output"]/name)).dataobj)
        self.assertFalse(result[0,0,0])
        self.assertTrue(result[2:4,2:4,2:4].all())


if __name__ == "__main__":
    unittest.main()
