"""Inference-only Detectron2 compatibility surface for BiomedParse.

The released BiomedParse checkpoint uses a small set of Detectron2 Python
wrappers but no compiled Detectron2 operators in its frozen inference graph.
Keeping that surface local avoids compiling the full training framework in the
CUDA-11.6 challenge image.  The implementation is intentionally minimal and
must remain covered by the real-model equivalence gate before deployment.
"""

