from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.article9_engine import Article9Engine
from src.article9_engine.normalization import normalize_text, root_form


class NormalizationTests(unittest.TestCase):
    def test_normalize_text_removes_accents_and_lowercases(self) -> None:
        value = normalize_text("Personne TRES croyante, tres engagee.")
        self.assertEqual(value, "personne tres croyante tres engagee")

    def test_root_form_handles_close_variants(self) -> None:
        self.assertEqual(root_form("croyance"), root_form("croyante"))


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = Article9Engine(config_path="configs/article9_categories.yml")

    def test_religion_detection_in_conseil_field(self) -> None:
        text = "Champ conseil : personne tres croyante et impliquee dans sa paroisse."
        result = self.engine.analyze_text(text, text_id="religion_case")
        religion = next(
            category
            for category in result.category_results
            if category.category_key == "RELIGION_BELIEF"
        )
        self.assertIn(religion.decision, {"review", "alert"})
        self.assertGreaterEqual(religion.score, 0.45)

    def test_health_detection_in_free_text(self) -> None:
        text = "Observation : antecedents medicaux et traitement en cours pour diabete."
        result = self.engine.analyze_text(text, text_id="health_case")
        health = next(
            category
            for category in result.category_results
            if category.category_key == "HEALTH_DATA"
        )
        self.assertIn(health.decision, {"review", "alert"})

    def test_clear_when_no_sensitive_signal(self) -> None:
        text = "Le client souhaite une gestion prudente et un versement programme."
        result = self.engine.analyze_text(text, text_id="clear_case")
        active = [category for category in result.category_results if category.decision != "clear"]
        self.assertEqual(active, [])

    def test_default_engine_declares_network_disabled_for_semantic(self) -> None:
        result = self.engine.analyze_text("Texte neutre", text_id="semantic_status_case")
        self.assertIn("semantic:network_disabled", result.decision_log)

    def test_engine_disables_semantic_when_local_model_is_missing(self) -> None:
        config = yaml.safe_load(Path("configs/article9_categories.yml").read_text(encoding="utf-8"))
        config["semantic"]["enabled"] = True
        config["semantic"]["allow_network_download"] = False
        config["semantic"]["local_files_only"] = True
        config["semantic"]["model_name"] = "sentence-transformers/__model_that_should_not_exist__"

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yml", delete=False) as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
            temp_config_path = handle.name

        try:
            engine = Article9Engine(config_path=temp_config_path)
            result = engine.analyze_text("Texte neutre", text_id="missing_local_model")
            self.assertIn("semantic:network_disabled", result.decision_log)
            self.assertIn("semantic:disabled_missing_local_model", result.decision_log)
        finally:
            Path(temp_config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
