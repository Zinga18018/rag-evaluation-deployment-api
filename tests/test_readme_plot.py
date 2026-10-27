"""Source/artifact checks use stdlib; rerendering alone needs optional Matplotlib."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from scripts.plot_development_results import (
    CATEGORIES, ROOT, SOURCE, SOURCE_SHA256, STEM, load_snapshot, provenance,
)

FIGURES = ROOT / "docs" / "figures"
EXPECTED = {
    "original_answerable_fixture": (20, 19, 1),
    "additional_answerable": (12, 12, 0),
    "overlapping_vocabulary_unanswerable": (12, 8, 4),
    "out_of_domain_unanswerable": (8, 0, 8),
}


class ReadmePlotTests(unittest.TestCase):
    def test_snapshot_hash_and_category_counts(self):
        self.assertEqual(sha256(SOURCE.read_bytes()).hexdigest(), SOURCE_SHA256)
        report = load_snapshot()
        self.assertEqual(report["protocol"]["label_status_counts"], {"pending_human_review": 52})
        self.assertEqual(set(EXPECTED), {key for key, _ in CATEGORIES})
        for key, (total, answered, abstained) in EXPECTED.items():
            group = report["category_counts"][key]
            self.assertEqual((group["cases"], group["answered"], group["abstained"]),
                             (total, answered, abstained))
            cases = [case for case in report["case_results"] if case["category"] == key]
            self.assertEqual(len(cases), total)
            counts = Counter(case["status"] for case in cases)
            self.assertEqual((counts["answered"], counts["abstained"]), (answered, abstained))
        false_answers = [case for case in report["case_results"]
                         if not case["expected_answerable"] and case["status"] == "answered"]
        self.assertEqual(len(false_answers), 8)
        self.assertTrue(all(case["category"] == "overlapping_vocabulary_unanswerable"
                            for case in false_answers))

    def test_changed_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_bytes(SOURCE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "snapshot hash changed"):
                load_snapshot(changed)

    def test_provenance_matches_source(self):
        saved = json.loads((FIGURES / f"{STEM}.provenance.json").read_text())
        self.assertEqual(saved, provenance(load_snapshot()))
        self.assertIn("not answer correctness", saved["interpretation"])
        self.assertIn("pending_human_review", saved["label_status_counts"])

    def test_svg_is_standalone_and_has_caveats(self):
        raw = (FIGURES / f"{STEM}.svg").read_bytes()
        svg = ET.fromstring(raw)
        self.assertTrue(svg.tag.endswith("}svg"))
        self.assertEqual(len(svg.attrib["viewBox"].split()), 4)
        forbidden = {"script", "foreignObject", "image", "iframe"}
        for node in svg.iter():
            self.assertNotIn(node.tag.rsplit("}", 1)[-1], forbidden)
            for key, value in node.attrib.items():
                self.assertFalse(key.lower().startswith("on"), key)
                if key.endswith("href"):
                    self.assertTrue(value.startswith("#"), value)
        text = " ".join(svg.itertext())
        for phrase in ("8 false answers", "pending human review", "not answer correctness",
                       "19", "12", "Abstained", "Answered"):
            self.assertIn(phrase, text)
        self.assertNotIn(b"<dc:date>", raw)

    def test_png_signature_and_dimensions(self):
        raw = (FIGURES / f"{STEM}.png").read_bytes()
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(raw[12:16], b"IHDR")
        width, height = struct.unpack(">II", raw[16:24])
        self.assertEqual((width, height), (979, 604))

    def test_readme_preserves_limitations_and_mirror_disclosure(self):
        readme = (ROOT / "README.md").read_text()
        for phrase in ("pending_human_review", "mirrored retriever and evaluator",
                       "not answer correctness", "requirements-plot.txt", f"{STEM}.svg"):
            self.assertIn(phrase, readme)
        self.assertIn("```mermaid", (ROOT / "WORKFLOW.md").read_text())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional requirements-plot.txt not installed")
    def test_optional_rerender_is_deterministic(self):
        original = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            env = dict(os.environ, MPLCONFIGDIR=str(tmp / "mplconfig"), PYTHONDONTWRITEBYTECODE="1")
            for name in ("first", "second"):
                subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_development_results.py"),
                                "--output-dir", str(tmp / name)], cwd=ROOT, env=env,
                               check=True, timeout=60, capture_output=True)
            for suffix in ("svg", "png", "provenance.json"):
                filename = f"{STEM}.{suffix}"
                first = (tmp / "first" / filename).read_bytes()
                self.assertEqual(first, (tmp / "second" / filename).read_bytes())
                # Cross-platform/font byte identity with historical artifacts is not promised.
        self.assertEqual(SOURCE.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
