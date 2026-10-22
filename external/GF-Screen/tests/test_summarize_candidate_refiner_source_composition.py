import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from summarize_candidate_refiner_oof import validate_source_composition_result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(path: Path, epoch: int, index_digest: str) -> dict:
    payload = {
        "fold": 0,
        "folds": 5,
        "epoch": epoch,
        "index_sha256": index_digest,
        "config": {"seed": 20260823},
        "model": {},
    }
    torch.save(payload, path)
    return payload


def identity(path: Path, payload: dict) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha(path),
        "fold": payload["fold"],
        "folds": payload["folds"],
        "seed": payload["config"]["seed"],
        "epoch": payload["epoch"],
        "index_sha256": payload["index_sha256"],
    }


class SourceCompositionAuditTests(unittest.TestCase):
    def test_every_route_checkpoint_is_bound_to_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_digest = "a" * 64
            class_path = root / "class.pt"
            mature_path = root / "mature.pt"
            discovery_path = root / "discovery.pt"
            class_payload = checkpoint(class_path, 10, index_digest)
            mature_payload = checkpoint(mature_path, 7, index_digest)
            discovery_payload = checkpoint(discovery_path, 10, index_digest)
            classification = root / "class.npz"
            mature = root / "mature.npz"
            discovery = root / "discovery.npz"
            output = root / "output.npz"
            for path, content in (
                (classification, b"classification"),
                (mature, b"mature"),
                (discovery, b"discovery"),
                (output, b"output"),
            ):
                path.write_bytes(content)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "index_sha256": index_digest,
                        "classification": str(classification.resolve()),
                        "classification_sha256": sha(classification),
                        "classification_checkpoint": identity(
                            class_path, class_payload
                        ),
                        "contour_routes": {
                            "mature": {
                                "sources": ["epoch105", "currentgf"],
                                "contour": str(mature.resolve()),
                                "contour_sha256": sha(mature),
                                "contour_checkpoint": identity(
                                    mature_path, mature_payload
                                ),
                            },
                            "discovery": {
                                "sources": ["autopet_weak", "lung_cell0"],
                                "contour": str(discovery.resolve()),
                                "contour_sha256": sha(discovery),
                                "contour_checkpoint": identity(
                                    discovery_path, discovery_payload
                                ),
                            },
                        },
                        "source_to_route": {
                            "epoch105": "mature",
                            "currentgf": "mature",
                            "autopet_weak": "discovery",
                            "lung_cell0": "discovery",
                        },
                        "output": str(output.resolve()),
                        "output_sha256": sha(output),
                        "labels_read_for_identity_only": True,
                        "labels_used_for_routing": False,
                        "public_read": False,
                        "hidden_read": False,
                    }
                ),
                encoding="utf-8",
            )
            result = validate_source_composition_result(
                audit,
                index_digest,
                class_path,
                class_payload,
                output,
            )
            self.assertEqual(
                set(result["validated_route_checkpoint_identities"]),
                {"mature", "discovery"},
            )

            discovery.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "routes_valid"):
                validate_source_composition_result(
                    audit,
                    index_digest,
                    class_path,
                    class_payload,
                    output,
                )


if __name__ == "__main__":
    unittest.main()
