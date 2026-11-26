import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import embed


class FakeCuda:
    def __init__(self, available, capability=(6, 1), architectures=()):
        self._available = available
        self._capability = capability
        self._architectures = list(architectures)

    def is_available(self):
        return self._available

    def get_device_capability(self, index):
        return self._capability

    def get_arch_list(self):
        return self._architectures


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda

    @staticmethod
    def device(name):
        return name


class EmbedTests(unittest.TestCase):
    def test_overlapping_windows_are_c_terminally_anchored(self):
        records = [("long", "A" * 1180)]
        windows = embed.make_windows(records, 1000, 128)
        self.assertEqual([item[2] for item in windows], [0, 180])
        self.assertEqual([item[0] for item in windows], [1000, 1000])

    def test_classifier_windows_are_non_overlapping(self):
        records = [("long", "A" * 1180)]
        windows = embed.make_windows(records, 1000, 0)
        self.assertEqual([(item[2], item[0]) for item in windows], [(1000, 180), (0, 1000)])

    def test_weighted_pooling_matches_whole_sequence_mean(self):
        residues = np.arange(1180 * 3, dtype=np.float32).reshape(1180, 3)
        records = [("long", "A" * len(residues))]
        total = np.zeros(3, dtype=np.float32)
        weight = 0
        for length, _, start, _ in embed.make_windows(records, 1000, 0):
            mean = residues[start:start + length].mean(axis=0)
            total += mean * length
            weight += length
        np.testing.assert_allclose(total / weight, residues.mean(axis=0), rtol=1e-6)

    def test_bad_records_are_reported_without_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.faa"
            path.write_text(
                ">good\nACDE\n>internal\nAC.DE\n>good\nACDE\n>terminal\nACDE*\n"
            )
            rejected = []
            records = list(embed.read_fasta(path, rejected=rejected))
        self.assertEqual(records, [("good", "ACDE"), ("terminal", "ACDE")])
        self.assertEqual([row["reason"] for row in rejected], ["internal_stop", "duplicate_id"])

    def test_auto_rejects_incompatible_cuda_architecture(self):
        torch = FakeTorch(FakeCuda(True, (6, 1), ("sm_70", "sm_80")))
        usable, reason = embed.cuda_compatibility(torch)
        self.assertFalse(usable)
        self.assertIn("6.1", reason)
        self.assertEqual(embed.choose_device(torch, "auto"), "cpu")

    def test_ptx_forward_compatibility_is_accepted(self):
        torch = FakeTorch(FakeCuda(True, (8, 6), ("sm_80", "compute_80")))
        self.assertTrue(embed.cuda_compatibility(torch)[0])
        self.assertEqual(embed.choose_device(torch, "auto"), "cuda:0")

    def test_retaining_residues_does_not_change_protein_embeddings(self):
        def fake_backend(name, torch, device):
            def encode(items):
                output = []
                for length, _, start, sequence in items:
                    # Deliberately context-dependent vectors: the residue pass
                    # differs, while the frozen classifier pass must not.
                    base = (start + length + sum(map(ord, sequence))) % 97
                    output.append(np.full((length, 1024), base, dtype=np.float32))
                return output
            return object(), encode, {"checkpoint": "fake"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "input.faa"
            fasta.write_text(">short\n" + "A" * 20 + "\n>long\n" + "C" * 1180 + "\n")
            outputs = []
            for skip in (False, True):
                output = root / ("skip" if skip else "retain")
                args = SimpleNamespace(
                    fasta=fasta, output=output, model="prott5", device="cpu",
                    max_batch_tokens=4000, max_batch_size=8,
                    skip_residue_embeddings=skip, overwrite=False,
                )
                with mock.patch.object(embed, "parse_args", return_value=args), \
                     mock.patch.object(embed, "load_backend", side_effect=fake_backend):
                    embed.main()
                outputs.append(np.load(output / "embeddings.npy"))
            np.testing.assert_array_equal(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
