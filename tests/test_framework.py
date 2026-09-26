"""Phase 1 tests: Algorithm 1, SafetyGovernor, MER, R_lang, YAML/CLI, workdir, Distiller."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beni.core import Analysis, Morpheme, Sentence, Token, sentence_to_completion_json
from beni.core.compute.metrics import MorphologyScorer
from beni.core.compute.rewards import RewardManager
from beni.core.language import MULTI13, Language, experiment_group_code, resolve_scope
from beni.core.safety.governor import SafetyGovernor, SafetySnapshot
from beni.core.safety.spec import SafetySpec
from beni.core.srl.algorithm1 import (
    maybe_distill_batch,
    maybe_distill_languages,
    batch_user_texts,
    batch_text_langs,
    group_texts_by_language,
)
from beni.core.srl.config import MasterConfig, RewardConfig
from beni.core.morphotactic.distil.distillation import DistillProposal, Distiller
from beni.utils import config as cfg


class TestLanguage:
    def test_mku_not_mlq(self):
        lang = Language.from_code("mku")
        assert lang.group_code == "mku"
        aliased = Language.from_code("mlq")
        assert aliased.group_code == "mku"

    def test_group_codes_dedupes_aliases(self):
        assert Language.group_codes(["bam", "bm", "mku", "mlq"]) == ["bam", "mku"]

    def test_packaged_alias(self):
        path = cfg.resolve_packaged_baseline_dir("mku")
        assert path is not None
        assert path.name in {"mku", "mlq"}
        aliased = cfg.resolve_packaged_baseline_dir("mlq")
        assert aliased is not None

    def test_multi13_and_outlier_scopes(self):
        multi = resolve_scope("multi13")
        assert multi.label == "MULTI13"
        assert multi.languages == MULTI13
        assert resolve_scope("all").languages == MULTI13
        assert resolve_scope("bam,mku").languages == ["bam", "mku"]
        assert resolve_scope("bam").label == "SINGLE_LANG"
        assert resolve_scope("bbo").label == "OUTLIER"
        mixed = resolve_scope(["bam", "bbo"])
        assert mixed.languages == ["bam"]
        assert mixed.dropped == ["bbo"]
        assert mixed.label == "SINGLE_LANG"

    def test_experiment_remap_does_not_change_global_mlq(self):
        assert Language.from_code("mlq").group_code == "mku"
        assert experiment_group_code("mlq") == "kao"
        assert experiment_group_code("hsy") == "mey"
        assert experiment_group_code("seq") == "spp"
        assert experiment_group_code("bbo") == "bbo"


class TestExperimentSplit:
    def test_bbo_dropped_from_multi13_and_test_json_refused(self, tmp_path):
        import json

        from beni.data.datasets import assert_train_source, load_experiment_records

        path = tmp_path / "samples.jsonl"
        rows = [
            {"text": "kao row", "lang": "mlq"},
            {"text": "mey row", "lang": "hsy"},
            {"text": "spp row", "lang": "seq"},
            {"text": "bam row", "lang": "bam"},
            {"text": "outlier", "lang": "bbo"},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        loaded = load_experiment_records(["multi13"], path)
        assert {row["lang"] for row in loaded["records"]} == {"kao", "mey", "spp", "bam"}
        assert loaded["dropped_bbo"] == 1
        heldout = tmp_path / "test.json"
        heldout.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="evaluation-only"):
            assert_train_source(heldout)

    def test_sft_is_registered_and_jax_is_rejected(self):
        from beni.core.srl.config import MasterConfig
        from beni.core.srl.sft.sft import SebeniSft
        from beni.core.srl.unified import SRLTrainer, get_algorithm

        assert get_algorithm("sft") is SebeniSft
        config = MasterConfig(algorithm="sft")
        config.trainer.framework = "jax"
        with pytest.raises(ValueError, match="torch-only"):
            SRLTrainer(config)


class TestSafetyGovernor:
    def test_promote_requires_phi_improve(self):
        gov = SafetyGovernor(SafetySpec())
        denied = gov.allow_promote(0.4, 0.4, parseable=True, first_create=False)
        assert denied.allowed is False
        assert denied.reason == "phi_not_improved"

        ok = gov.allow_promote(0.4, 0.6, parseable=True, first_create=False)
        assert ok.allowed is True

    def test_first_create_parse_gate(self):
        gov = SafetyGovernor(SafetySpec())
        denied = gov.allow_promote(0.0, 0.0, parseable=False, first_create=True)
        assert denied.allowed is False
        assert denied.reason == "unparseable_bootstrap"
        ok = gov.allow_promote(0.0, 0.0, parseable=True, first_create=True)
        assert ok.allowed is True

    def test_format_invalid_blocks_update(self):
        gov = SafetyGovernor(SafetySpec())
        assert gov.allow_policy_update({"format_scores": [0.0, 0.0]}) is False
        assert gov.allow_policy_update({"format_scores": [0.2, 0.2]}) is True

    def test_hub_requires_card_and_snapshot(self, tmp_path):
        gov = SafetyGovernor(SafetySpec())
        assert gov.allow_hub_push(tmp_path) is False
        (tmp_path / "README.md").write_text("# card\n", encoding="utf-8")
        gov.record_snapshot(phi=0.7, tau=0.5, checkpoint_id="baseline_v1")
        gov.write_snapshot(tmp_path)
        assert gov.allow_hub_push(tmp_path) is True


class TestAlgorithm1:
    def _distiller(self, phi, proposal):
        distiller = MagicMock()
        distiller.phi_on_texts.return_value = phi
        distiller.propose.return_value = proposal
        distiller.is_first_create.return_value = False
        return distiller

    def test_phi_below_tau_triggers_distiller(self):
        proposal = DistillProposal("g", "d", 0.2, 0.8, True, False)
        distiller = self._distiller(0.2, proposal)
        gov = SafetyGovernor(SafetySpec())
        decision = maybe_distill_batch(["aw ka ne labato."], distiller, gov, tau=0.5)
        distiller.propose.assert_called_once()
        distiller.write_checkpoint.assert_called_once_with("g", "d")
        assert decision.allowed is True

    def test_phi_prime_not_greater_does_not_promote(self):
        proposal = DistillProposal("g", "d", 0.2, 0.2, True, False)
        distiller = self._distiller(0.2, proposal)
        gov = SafetyGovernor(SafetySpec())
        decision = maybe_distill_batch(["text"], distiller, gov, tau=0.5)
        distiller.propose.assert_called_once()
        distiller.write_checkpoint.assert_not_called()
        assert decision.allowed is False
        assert decision.reason == "phi_not_improved"

    def test_phi_at_or_above_tau_skips_llm_ref(self):
        distiller = self._distiller(0.9, None)
        gov = SafetyGovernor(SafetySpec())
        decision = maybe_distill_batch(["text"], distiller, gov, tau=0.5)
        distiller.propose.assert_not_called()
        distiller.write_checkpoint.assert_not_called()
        assert decision.reason == "phi_above_tau"

    def test_batch_user_texts_from_chat(self):
        texts = batch_user_texts({
            "prompt": [[{"role": "system", "content": "x"}, {"role": "user", "content": "aw ka"}]]
        })
        assert texts == ["aw ka"]

    def test_mixed_batch_keeps_per_row_language(self):
        pairs = batch_text_langs({
            "prompt": [
                [{"role": "user", "content": "aw ka"}],
                [{"role": "user", "content": "i ni ce"}],
            ],
            "language": ["bam", "mku"],
        })
        assert pairs == [("aw ka", "bam"), ("i ni ce", "mku")]
        grouped = group_texts_by_language(pairs)
        assert grouped["bam"] == ["aw ka"]
        assert grouped["mku"] == ["i ni ce"]

    def test_maybe_distill_languages_calls_each_group(self):
        calls = []

        def distiller_for(group):
            d = MagicMock()
            d.phi_on_texts.return_value = 0.9
            d.propose.return_value = None
            d.is_first_create.return_value = False
            calls.append(group)
            return d

        gov = SafetyGovernor(SafetySpec())
        decisions = maybe_distill_languages(
            [("aw ka", "bam"), ("i ni ce", "mku")],
            distiller_for,
            gov,
            tau=0.5,
        )
        assert set(calls) == {"bam", "mku"}
        assert set(decisions) == {"bam", "mku"}
        assert all(d.reason == "phi_above_tau" for d in decisions.values())


class TestMER:
    def test_mer_identical(self):
        scorer = MorphologyScorer()
        assert scorer.mer(["a", "b"], ["a", "b"]) == 0.0

    def test_mer_substitution(self):
        scorer = MorphologyScorer()
        assert scorer.mer(["a", "b"], ["a", "c"]) == pytest.approx(0.5)

    def test_mer_sentence(self):
        scorer = MorphologyScorer()
        ref = Sentence(text="x", tokens=[Token(surface="ab", stage=1)])
        hyp = Sentence(text="x", tokens=[Token(surface="ab", stage=1)])
        ref.tokens[0].analyses = []
        assert scorer.mer_sentence(ref, hyp) == 0.0


class TestEvalCosts:
    def test_uwec_absolute_log_ratio_is_a_cost(self):
        from beni.core.compute.uwec import uwec

        stages = [1, 1]
        _, high = uwec(stages, [0.9, 0.9], [0.1, 0.1], beta=0.1, eps=1e-8)
        _, low = uwec(stages, [0.1, 0.1], [0.9, 0.9], beta=0.1, eps=1e-8)
        assert float(high) == pytest.approx(float(low))
        assert float(high) > 1.0

    def test_uwec_stage_minus_one_is_zero_indicator(self):
        from beni.core.compute.uwec import uwec

        _, unknown = uwec([-1], [0.5], [0.5], beta=0.1, eps=1e-8)
        _, known = uwec([1], [0.5], [0.5], beta=0.1, eps=1e-8)
        assert float(unknown) == pytest.approx(0.0)
        assert float(known) == pytest.approx(1.0)

    def test_mer_pools_by_morpheme_count(self):
        from beni.core.compute.helpers import mer_edit_counts, mer_micro

        ops_a, n_a = mer_edit_counts(["a"], ["b"])
        ops_b, n_b = mer_edit_counts(["x", "y", "z"], ["x", "y", "z"])
        pooled = (int(ops_a) + int(ops_b)) / (int(n_a) + int(n_b))
        mean_of_rates = (float(mer_micro(["a"], ["b"])) + float(mer_micro(["x", "y", "z"], ["x", "y", "z"]))) / 2
        assert pooled == pytest.approx(0.25)
        assert mean_of_rates == pytest.approx(0.5)

    def test_reported_mcs_is_mismatch_reward_mcs_is_match(self):
        from beni.core.compute.helpers import mcs_mismatch

        scorer = MorphologyScorer()
        pred = Sentence(text="x", tokens=[Token(surface="a", stage=1), Token(surface="b", stage=-1)])
        ref = Sentence(text="x", tokens=[Token(surface="a", stage=1), Token(surface="b", stage=1)])
        assert scorer.mcs(pred, ref) == pytest.approx(0.5)
        assert float(mcs_mismatch([1, -1], [1, 1])) == pytest.approx(0.5)
        extra = float(mcs_mismatch([1], [1, 2]))
        assert extra == pytest.approx(0.5)

    def test_numpy_and_jax_match_when_jax_is_installed(self):
        numpy = pytest.importorskip("numpy")
        from beni.core.compute.helpers import mer_micro, mcs_mismatch
        from beni.core.compute.uwec import uwec

        stages = [1, -1, 2]
        theta = [0.2, 0.4, 0.8]
        ref = [0.5, 0.5, 0.1]
        _, np_uwec = uwec(stages, theta, ref, xp=numpy)
        np_mer = mer_micro(["a", "b"], ["a", "c"], xp=numpy)
        np_mcs = mcs_mismatch([1, 2], [1, 3], xp=numpy)
        try:
            import jax.numpy as jnp
        except Exception:
            pytest.skip("jax is not installed")
        _, jax_uwec = uwec(stages, theta, ref, xp=jnp)
        jax_mer = mer_micro(["a", "b"], ["a", "c"], xp=jnp)
        jax_mcs = mcs_mismatch([1, 2], [1, 3], xp=jnp)
        assert float(np_uwec) == pytest.approx(float(jax_uwec), rel=1e-6)
        assert float(np_mer) == pytest.approx(float(jax_mer))
        assert float(np_mcs) == pytest.approx(float(jax_mcs))


class TestAdjustedPipeline:
    def test_sentence_to_completion_json(self):
        sentence = Sentence(
            text="aw",
            lang="bam",
            tokens=[
                Token(
                    surface="aw",
                    stage=1,
                    analyses=[
                        Analysis(
                            form="áw",
                            ps=["prn"],
                            morphemes=[Morpheme(form="áw", ps=["prn"])],
                        )
                    ],
                )
            ],
        )
        payload = sentence_to_completion_json(sentence)
        assert payload["lang"] == "bam"
        assert payload["tokens"][0]["analyses"][0]["morphemes"][0]["form"] == "áw"

    def test_algorithmic_distiller_adds_misses_without_provider(self, tmp_path):
        with patch.object(
            Distiller, "_valid_language", return_value={"language": "Z", "group_code": "zzz"}
        ):
            distiller = Distiller(
                lang_code="zzz", backend="algorithmic", working_dir=tmp_path
            )
        assert distiller.provider is None
        with patch.object(distiller, "collect_misses", return_value=["foo"]), patch.object(
            distiller, "files_parseable", return_value=True
        ), patch.object(distiller, "phi_on_texts", return_value=1.0):
            proposal = distiller.propose(["foo"], current_phi=0.0)
        assert proposal is not None
        assert "\\lx foo" in proposal.dict_text
        assert proposal.phi_prime == 1.0

    def test_algorithmic_scratch_phi_improves(self, tmp_path):
        pytest.importorskip("daba.mparser")
        with patch.object(
            Distiller, "_valid_language", return_value={"language": "Z", "group_code": "zzz"}
        ), patch(
            "beni.core.morphotactic.distil.distillation.cfg.resolve_packaged_baseline_dir",
            return_value=None,
        ):
            distiller = Distiller(
                lang_code="zzz", backend="algorithmic", working_dir=tmp_path
            )
            distiller.handle_baselines()
            phi = distiller.phi_on_texts(["foobarqux"])
            proposal = distiller.propose(["foobarqux"], current_phi=phi)
        assert proposal is not None
        assert proposal.parseable
        assert proposal.phi_prime > phi

    def test_morph_reward_compares_gold_annotation(self):
        reference = {
            "text": "aw",
            "lang": "bam",
            "tokens": [
                {
                    "surface": "aw",
                    "stage": 1,
                    "analyses": [
                        {
                            "form": "áw",
                            "ps": ["prn"],
                            "gloss": "",
                            "morphemes": [{"form": "áw", "ps": ["prn"]}],
                        }
                    ],
                }
            ],
        }
        rm = RewardManager()
        score = rm.reward_morph(
            [__import__("json").dumps(reference)],
            language=["bam"],
            reference=[reference],
            prompts=[{}],
        )
        assert score == pytest.approx([rm.config.morph_weight])

    def test_reward_format_accepts_conversational_completions(self):
        from beni.core.compute.rewards import RewardManager

        blob = '{"lang": "bam", "tokens": []}'
        rm = RewardManager()
        conversational = [[{"role": "assistant", "content": blob}]]
        assert rm.completion_text(conversational[0]) == blob
        scores = rm.reward_format(conversational, language=["bam"])
        assert scores == [rm.config.format_weight]
        assert rm.extract_json(conversational[0])["lang"] == "bam"

    def test_preference_fallback_uses_gold_and_nonempty_negative(self):
        from beni.data.datasets import rank_group_to_preference

        reference = {
            "text": "aw",
            "lang": "bam",
            "tokens": [{"surface": "aw", "stage": 1, "analyses": []}],
        }
        dataset = rank_group_to_preference(
            [{"text": "aw", "lang": "bam", "reference": reference}]
        )
        assert dataset[0]["chosen"] != "aw"
        assert dataset[0]["rejected"]
        assert dataset[0]["chosen"] != dataset[0]["rejected"]

    def test_uncertainty_uses_stage_indicator_and_never_amplifies(self):
        from beni.core.srl.jax import jax_uncertainty_scale

        scale, values = jax_uncertainty_scale(
            [[1, -1]], model_logps=[-1.0], ref_logps=[-2.0], beta=0.1
        )
        assert values["u_indicator"] == 0.5
        assert values["u_kl"] == pytest.approx(0.1)
        assert scale == pytest.approx(1.0 / 1.6)

        scale, _ = jax_uncertainty_scale(
            [], model_logps=[-3.0], ref_logps=[-1.0], beta=1.0
        )
        assert scale == 1.0

    def test_safety_governor_uses_full_uncertainty(self):
        gov = SafetyGovernor(SafetySpec(kl_beta=0.1))
        meta = {
            "predicted_stages": [[1, -1]],
            "model_logps": [-1.0],
            "ref_logps": [-2.0],
        }
        assert gov._uncertainty_scale(meta) == pytest.approx(1.0 / 1.6)
        assert meta["u_indicator"] == 0.5

    def test_jax_selection_and_clear_torch_fallback(self, capsys):
        from beni.core.srl.jax import JaxPolicyPlugin
        from beni.core.srl.grpo.grpo import SebeniGrpo
        from beni.core.srl.unified import SRLTrainer

        config = MasterConfig()
        config.trainer.framework = "jax"
        driver = SRLTrainer(config)
        assert isinstance(driver.plugin, JaxPolicyPlugin)

        with patch.object(SebeniGrpo, "load_models", return_value=MagicMock()):
            with pytest.warns(RuntimeWarning, match="Continuing with PyTorch/TRL"):
                driver.plugin._fallback_to_torch(RuntimeError("no flax_model.msgpack"))
        assert config.trainer.framework == "torch"
        assert "Flax weights" in capsys.readouterr().out


class TestRLang:
    def test_matching_lang(self):
        rm = RewardManager()
        completion = '{"text": "aw", "lang": "bam", "tokens": []}'
        scores = rm.reward_lang([completion], language=["bam"])
        assert scores == [0.1]

    def test_mismatch_zero(self):
        rm = RewardManager()
        completion = '{"text": "aw", "lang": "fra", "tokens": []}'
        scores = rm.reward_lang([completion], language=["bam"])
        assert scores == [0.0]


class TestWorkdirAndYaml:
    def test_temp_workdir_cli_wins(self, tmp_path):
        wd = cfg.resolve_working_dir(cli_path=tmp_path)
        assert wd.root == tmp_path.resolve()
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "models").is_dir()

    def test_yaml_load(self, tmp_path):
        yml = tmp_path / "cfg.yaml"
        yml.write_text(
            "algorithm: dpo\nworking_dir: run-local\ndistillation:\n  tau: 0.7\n  hitl: true\n",
            encoding="utf-8",
        )
        config = MasterConfig.from_yaml(yml)
        assert config.algorithm == "dpo"
        assert config.distillation.tau == 0.7
        assert config.distillation.hitl is True
        assert Path(config.working_dir) == (tmp_path / "run-local").resolve()

    def test_yaml_languages_and_hyperparams(self, tmp_path):
        yml = tmp_path / "cfg.yaml"
        yml.write_text(
            "\n".join([
                "algorithm: grpo",
                "working_dir: run-ml",
                "data:",
                "  languages: [bam, mku]",
                "  default_lang: bam",
                "trainer:",
                "  learning_rate: 1.0e-5",
                "  per_device_train_batch_size: 4",
                "  num_generations: 8",
                "  temperature: 0.7",
                "model:",
                "  lora_r: 32",
                "  lora_alpha: 64",
            ]) + "\n",
            encoding="utf-8",
        )
        config = MasterConfig.from_yaml(yml)
        assert config.languages() == ["bam", "mku"]
        assert config.trainer.learning_rate == 1e-5
        assert config.trainer.per_device_train_batch_size == 4
        assert config.trainer.num_generations == 8
        assert config.trainer.temperature == 0.7
        assert config.model.lora_r == 32
        assert config.model.lora_alpha == 64

    def test_cli_overrides(self):
        config = MasterConfig()
        config.apply_cli_overrides(
            languages=["bam", "mku,dtm"],
            learning_rate=2e-5,
            batch_size=8,
            lora_r=8,
            beta=0.04,
        )
        assert config.languages() == ["bam", "mku", "dtm"]
        assert config.trainer.learning_rate == 2e-5
        assert config.trainer.per_device_train_batch_size == 8
        assert config.trainer.beta == 0.04
        assert config.model.lora_r == 8


class TestScratchBootstrap:
    def test_writes_stubs_when_no_packaged(self, tmp_path):
        with patch(
            "beni.core.morphotactic.distil.distillation.create_provider",
            return_value=MagicMock(capability=MagicMock(value="none"), cache=None),
        ), patch.object(
            Distiller, "_valid_language", return_value={"language": "Z", "group_code": "zzz"}
        ):
            distiller = Distiller(lang_code="zzz", provider="openai", working_dir=tmp_path)
            distiller.handle_baselines()
        gram = tmp_path / "data" / "baselines" / "zzz" / "baseline.gram"
        assert gram.exists()
        assert "sebeni-scratch-bootstrap" in gram.read_text(encoding="utf-8")
        assert distiller.is_first_create() is True
        assert distiller.prompt_mode() == "bootstrap"


class TestProviderRegistry:
    def test_known_providers(self):
        from beni.core.morphotactic.distil.providers import PROVIDER_REGISTRY

        for name in ("google", "gemini", "openai", "groq", "together", "gguf"):
            assert name in PROVIDER_REGISTRY
        with patch(
            "beni.core.morphotactic.distil.providers.openai_compat.OpenAICompatibleProvider.__init__",
            return_value=None,
        ):
            from beni.core.morphotactic.distil.providers.openai_compat import OpenAIProvider

            p = OpenAIProvider("sk-test", "gpt-4o")
            assert p.__class__.__name__ == "OpenAIProvider"

    def test_unknown_provider(self):
        from beni.core.morphotactic.distil.providers import create_provider

        with pytest.raises(ValueError):
            create_provider("nope")

    def test_google_vertex_allows_adc_without_api_key(self):
        pytest.importorskip("google.genai")
        with patch(
            "beni.core.morphotactic.distil.providers.google.genai.Client"
        ) as client:
            from beni.core.morphotactic.distil.providers.google import GoogleProvider

            provider = GoogleProvider(
                api_key=None, vertex=True, project_id="test-project", language="bam"
            )
        assert provider.cache is None
        client.assert_called_with(
            vertexai=True, project="test-project", location=cfg.GOOGLE_LOCATION
        )

    def test_gguf_selection_warns_about_context(self):
        from beni.core.morphotactic.distil.providers.gguf import GGUFProvider

        with patch.object(GGUFProvider, "initialize", return_value=MagicMock()):
            with pytest.warns(RuntimeWarning, match="context"):
                GGUFProvider(gguf_path="/tmp/model.gguf", n_ctx=2048)

    def test_dotenv_does_not_override_process_env(self, tmp_path, monkeypatch):
        dotenv = tmp_path / ".env"
        dotenv.write_text("SEBENI_TEST_KEY=file\nNEW_TEST_KEY=value\n", encoding="utf-8")
        monkeypatch.setenv("SEBENI_TEST_KEY", "process")
        monkeypatch.delenv("NEW_TEST_KEY", raising=False)
        cfg.load_dotenv([dotenv])
        assert __import__("os").environ["SEBENI_TEST_KEY"] == "process"
        assert __import__("os").environ["NEW_TEST_KEY"] == "value"


class TestAlgorithmDispatch:
    def test_get_algorithm(self):
        from beni.core.srl.unified import get_algorithm
        from beni.core.srl.grpo.grpo import SebeniGrpo
        from beni.core.srl.dpo.dpo import SebeniDpo
        from beni.core.srl.apo.apo import SebeniApo

        assert get_algorithm("grpo") is SebeniGrpo
        assert get_algorithm("dpo") is SebeniDpo
        assert get_algorithm("apo") is SebeniApo

    def test_srl_trainer_is_not_grpo_alias(self):
        from beni.core.srl.unified import SRLTrainer
        from beni.core.srl.grpo.grpo import SebeniGrpo

        assert SRLTrainer is not SebeniGrpo
        trainer = SRLTrainer(MasterConfig(algorithm="grpo"))
        assert isinstance(trainer.plugin, SebeniGrpo)


class TestCLI:
    def test_init_and_help(self, tmp_path):
        from typer.testing import CliRunner
        from beni.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.stdout
        assert "exp" in result.stdout
        help_exp = runner.invoke(app, ["exp", "--help"])
        assert help_exp.exit_code == 0
        import re as _re

        exp_plain = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", help_exp.stdout)
        assert "packaged" in exp_plain.lower() or "raw" in exp_plain.lower()
        out = tmp_path / "run"
        result = runner.invoke(app, ["init", "--lang", "bam", "-w", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "config.yaml").exists()
        yaml_text = (out / "config.yaml").read_text(encoding="utf-8")
        assert "algorithm: grpo" in yaml_text
        assert "languages: [bam]" in yaml_text
        assert "learning_rate:" in yaml_text
        assert (out / "data").is_dir()

        multi = tmp_path / "multi"
        result = runner.invoke(
            app, ["init", "--lang", "bam", "--lang", "mku", "-w", str(multi)]
        )
        assert result.exit_code == 0, result.output
        multi_yaml = (multi / "config.yaml").read_text(encoding="utf-8")
        assert "languages: [bam, mku]" in multi_yaml

        help_train = runner.invoke(app, ["train", "--help"])
        assert help_train.exit_code == 0
        import re

        plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", help_train.stdout)
        assert "--lr" in plain
        assert "lora-r" in plain
        assert "--lang" in plain


class TestHeadlessDabaX:
    def test_wrong_pypi_daba_has_actionable_error(self):
        from beni.core.morphotactic.dabax import _daba_modules

        package_metadata = {
            "Summary": "daba by Klivolks",
            "Home-page": "https://github.com/klivolks/DaBa",
            "Author": "Vishnu Prakash",
        }
        real_import = __import__

        def import_without_mparser(name, *args, **kwargs):
            if name == "daba":
                raise ImportError("cannot import name 'mparser' from 'daba'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_mparser), patch(
            "beni.core.morphotactic.dabax.metadata.metadata",
            return_value=package_metadata,
        ):
            with pytest.raises(ImportError) as raised:
                _daba_modules()

        message = str(raised.value)
        assert "unrelated Klivolks Mongo helper" in message
        assert "pip uninstall -y daba" in message
        assert "github.com/maslinych/daba.git" in message
        assert "--no-deps" in message
        assert "wxPython is not required" in message
        assert "daba>=0.9.5" not in message

    def test_missing_runtime_dep_has_actionable_error(self):
        from beni.core.morphotactic.dabax import _daba_modules

        real_import = __import__

        def import_without_pkg_resources(name, *args, **kwargs):
            if name == "daba" or name.startswith("daba."):
                missing = ModuleNotFoundError("No module named 'pkg_resources'")
                missing.name = "pkg_resources"
                raise missing
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_pkg_resources), patch(
            "beni.core.morphotactic.dabax._wrong_pypi_daba", return_value=False
        ), patch(
            "beni.core.morphotactic.dabax._maslinych_parser_present", return_value=True
        ):
            with pytest.raises(ImportError) as raised:
                _daba_modules()

        message = str(raised.value)
        assert "CLI runtime dependency is missing" in message
        assert "pkg_resources" in message
        assert "pip install setuptools" in message
        assert "daba>=0.9.5" not in message
        assert "pip uninstall -y daba" not in message

    def test_get_dabax_reuses_instance_for_same_checkpoint(self, tmp_path):
        from beni.core.morphotactic import dabax as dabax_mod

        dabax_mod.clear_dabax_cache()
        gram = tmp_path / "baseline.gram"
        ldict = tmp_path / "baseline.dict"
        gram.write_text("x\n", encoding="utf-8")
        ldict.write_text("y\n", encoding="utf-8")
        runtime = tmp_path / "runtime"
        fake = MagicMock()
        with patch.object(dabax_mod, "DabaX", return_value=fake) as ctor:
            first = dabax_mod.get_dabax(
                "bam", gram=gram, ldict=ldict, runtime_dir=runtime
            )
            second = dabax_mod.get_dabax(
                "bam", gram=gram, ldict=ldict, runtime_dir=runtime
            )
            assert first is second is fake
            assert ctor.call_count == 1
            other = tmp_path / "other.dict"
            other.write_text("z\n", encoding="utf-8")
            third = dabax_mod.get_dabax(
                "bam", gram=gram, ldict=other, runtime_dir=runtime
            )
            assert third is fake
            assert ctor.call_count == 2
        dabax_mod.clear_dabax_cache()

    def test_build_dabax_reference_reuses_parser(self):
        from beni.core.morphotactic import dabax as dabax_mod
        from beni.data.datasets import build_dabax_reference

        dabax_mod.clear_dabax_cache()
        parser = MagicMock()
        parser.loader.return_value = []
        with patch.object(dabax_mod, "get_dabax", return_value=parser) as getter:
            build_dabax_reference("aw ka", "bam")
            build_dabax_reference("n be taa", "bam")
            assert getter.call_count == 2
            assert parser.loader.call_count == 2
        dabax_mod.clear_dabax_cache()

    def test_loader_without_wx(self, tmp_path):
        pytest.importorskip("daba.mparser")
        from beni.core.morphotactic.dabax import DabaX

        packaged = cfg.resolve_packaged_baseline_dir("bam")
        if packaged is None:
            pytest.skip("no packaged bam baseline")
        dabax = DabaX(
            "bam",
            gram=packaged / "baseline.gram",
            ldict=packaged / "baseline.dict",
            runtime_dir=tmp_path / "runtime",
        )
        sents = dabax.loader("a")
        assert isinstance(sents, list)


class TestModelCard:
    def test_writes_readme(self, tmp_path):
        from beni.core.hub.model_card import write_model_card

        snap = SafetySnapshot(phi=0.6, tau=0.5, checkpoint_id="baseline_v2", algorithm="grpo")
        path = write_model_card(tmp_path, snapshot=snap, config=MasterConfig())
        text = path.read_text(encoding="utf-8")
        assert "base_model" in text
        assert "0.6" in text
        assert "seben.robotsmali.org" in text
        assert "mlsftwrs.github.io/sebeni" in text


class TestMultilingualData:
    def test_loader_filters_to_configured_languages(self, tmp_path):
        from beni.data.datasets import SebeniDataLoader
        from beni.core.srl.config import DataConfig

        path = tmp_path / "mix.jsonl"
        path.write_text(
            '{"text": "aw ka", "lang": "bam"}\n'
            '{"text": "bonjour", "lang": "fra"}\n'
            '{"text": "i ni ce", "lang": "mku"}\n',
            encoding="utf-8",
        )
        loader = SebeniDataLoader(DataConfig(languages=["bam", "mku"], default_lang="bam"))
        recs = loader.load(path)
        langs = {r["lang"] for r in recs}
        assert langs == {"bam", "mku"}
        assert loader.skipped >= 1

    def test_prompt_names_all_languages(self):
        from beni.core.srl.config import SRLGrpoPrompt

        text = SRLGrpoPrompt(languages=["bam", "mku"]).prompt()
        assert "bam" in text
        assert "mku" in text
        assert "THIS sentence" in text


class TestPackagedSplits:
    def test_test_split_maps_mlq_to_mku(self):
        from beni.data.datasets import SebeniDataLoader

        loader = SebeniDataLoader()
        recs = loader.load_sebeni(split="test")
        langs = {r["lang"] for r in recs}
        assert recs
        assert "mku" in langs
        assert "mlq" not in langs
        assert "bam" in langs

    def test_train_from_raw_dir(self, tmp_path):
        from beni.data.datasets import SebeniDataLoader

        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "mlq.txt").write_text("i ni ce\n\nsecond\n", encoding="utf-8")
        (raw / "bam.txt").write_text("aw ka\n", encoding="utf-8")
        (raw / "notes.old").write_text("ignore", encoding="utf-8")
        loader = SebeniDataLoader()
        recs = loader.load_sebeni(data_dir=tmp_path, split="train")
        langs = {r["lang"] for r in recs}
        assert langs == {"mku", "bam"}
        texts = {r["text"] for r in recs}
        assert "i ni ce" in texts
        assert "aw ka" in texts


class TestWordfreqRawPipeline:
    def test_config_language_processes_raw_text(self, tmp_path):
        from beni.core.wordfreq import WordfreqReport, count_raw_inputs

        raw = tmp_path / "corpus.txt"
        raw.write_text("aw ka\nsecond line\n", encoding="utf-8")

        def fake_count(texts, lang, gram=None, ldict=None, checkpoint_id=None):
            values = list(texts)
            return WordfreqReport(
                language=lang,
                checkpoint_id=checkpoint_id,
                surfaces={values[0]: 1},
                n_sentences=len(values),
            )

        with patch(
            "beni.core.wordfreq.resolve_baseline_files",
            return_value=(Path("g"), Path("d"), "packaged"),
        ), patch("beni.core.wordfreq.count_texts", side_effect=fake_count):
            reports = count_raw_inputs(raw, languages=["bam"], default_lang="bam")
        assert set(reports) == {"bam"}
        assert reports["bam"].n_sentences == 2

    def test_cli_wordfreq_needs_no_llm_key(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from beni.cli.main import app

        raw = tmp_path / "bam.txt"
        raw.write_text("a\n", encoding="utf-8")
        config = tmp_path / "config.yaml"
        config.write_text(
            "\n".join(
                [
                    f"working_dir: {tmp_path / 'run'}",
                    "data:",
                    "  default_lang: bam",
                    "  languages: [bam]",
                    "wordfreq:",
                    f"  raw_inputs: {raw}",
                    "distillation:",
                    "  backend: algorithmic",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        result = CliRunner().invoke(app, ["wordfreq", "-c", str(config)])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "run" / "exp" / "wordfreq" / "bam" / "wordfreq.json").exists()


class TestExperimentConfig:
    def test_yaml_overlay(self, tmp_path):
        yml = tmp_path / "exp.yaml"
        yml.write_text(
            "\n".join([
                "algorithm: grpo",
                "working_dir: run-exp",
                "experiment:",
                "  kveritas: true",
                "  kveritas_seal: true",
                "trainer:",
                "  report_to: none",
            ]) + "\n",
            encoding="utf-8",
        )
        config = MasterConfig.from_yaml(yml)
        assert config.experiment.kveritas is True
        assert config.experiment.kveritas_seal is True
        assert config.trainer.report_to == "none"

    def test_kveritas_metrics(self, capsys):
        from beni.cli.main import emit_kveritas_metrics

        emit_kveritas_metrics({"phi": 0.51, "by_language": {"bam": {"phi": 0.6}}})
        out = capsys.readouterr().out
        assert "KVERITAS_METRIC name=phi value=0.51 step=0" in out
        assert "KVERITAS_METRIC name=phi_bam value=0.6 step=0" in out

    def test_jsonl_mlq_becomes_mku(self, tmp_path):
        from beni.data.datasets import SebeniDataLoader
        from beni.core.srl.config import DataConfig

        path = tmp_path / "mix.jsonl"
        path.write_text('{"text": "i ni ce", "lang": "mlq"}\n', encoding="utf-8")
        recs = SebeniDataLoader(DataConfig()).load(path)
        assert recs[0]["lang"] == "mku"

