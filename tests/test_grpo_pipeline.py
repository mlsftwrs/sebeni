import pytest
from typing import List
from unittest.mock import MagicMock, patch, ANY

from beni.core.srl import (
    MasterConfig, RewardConfig
)
from beni.core.compute.rewards import RewardManager
from beni.core.srl.grpo.grpo import SebeniGrpo

TEST_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "facebook/MobileLLM-350M",
]


class TestGRPOConfigs:
    """Test suite for GRPO pipeline configurations."""

    def test_default_master_config(self):
        config = MasterConfig()
        assert config.project_name == "GRPO-Morphology-Advanced"
        assert config.model.model_name == "HuggingFaceTB/SmolLM2-135M"
        assert config.distillation.enabled is True
        assert config.distillation.selected_backend == "algorithmic"
        assert config.reward.format_weight == 0.1
        assert config.reward.morph_weight == 0.4
        assert config.reward.rule_weight == 0.4
        assert config.reward.lang_weight == 0.1

    def test_master_config_from_dict(self):
        config_dict = {
            "project_name": "Custom-Morph-Experiment",
            "model": {"model_name": "facebook/MobileLLM-350M", "load_in_4bit": True},
            "distillation": {"enabled": False, "provider": "gemini", "model": "gemini-2.5-flash"},
            "reward": {"format_weight": 0.5, "morph_weight": 0.5, "enable_rule_reward": False}
        }
        config = MasterConfig.from_dict(config_dict)
        assert config.project_name == "Custom-Morph-Experiment"
        assert config.model.model_name == "facebook/MobileLLM-350M"
        assert config.distillation.enabled is False
        assert config.distillation.provider == "gemini"
        assert config.reward.format_weight == 0.5
        assert config.reward.enable_rule_reward is False

    def test_srl_trainer_is_facade(self):
        from beni.core.srl.unified import SRLTrainer
        from beni.core.srl.grpo.grpo import SebeniGrpo
        assert SRLTrainer is not SebeniGrpo
        trainer = SRLTrainer()
        assert isinstance(trainer.plugin, SebeniGrpo)


class TestTrlConfigKwargs:
    """Filter Sebeni trainer dicts to the installed TRL constructor."""

    def test_drops_unknown_grpo_fields(self):
        from beni.core.srl.config import GRPOTrainerConfig, trl_config_kwargs

        class FakeGRPO:
            def __init__(self, max_completion_length=1, warmup_steps=0, beta=0.1):
                pass

        out = trl_config_kwargs(FakeGRPO, GRPOTrainerConfig().to_dict())
        assert "max_prompt_length" not in out
        assert "warmup_ratio" not in out
        assert out["max_completion_length"] == 1024
        assert out["warmup_steps"] == 0
        assert out["beta"] == 0.1

    def test_keeps_legacy_grpo_fields(self):
        from beni.core.srl.config import GRPOTrainerConfig, trl_config_kwargs

        class FakeGRPO:
            def __init__(
                self,
                max_prompt_length=1,
                warmup_ratio=0.0,
                max_completion_length=1,
            ):
                pass

        out = trl_config_kwargs(FakeGRPO, GRPOTrainerConfig().to_dict())
        assert out["max_prompt_length"] == 1024
        assert out["warmup_ratio"] == 0.0
        assert out["max_completion_length"] == 1024

    def test_dpo_aliases_prompt_length_when_needed(self):
        from beni.core.srl.config import trl_config_kwargs

        class FakeDPO:
            def __init__(self, max_length=1, beta=0.1):
                pass

        out = trl_config_kwargs(
            FakeDPO,
            {"max_prompt_length": 512, "beta": 0.2},
            aliases={"max_prompt_length": "max_length"},
        )
        assert out == {"max_length": 512, "beta": 0.2}

    def test_dpo_keeps_explicit_max_length(self):
        from beni.core.srl.config import DPOTrainerConfig, trl_config_kwargs

        class FakeDPO:
            def __init__(self, max_length=1, beta=0.1):
                pass

        out = trl_config_kwargs(
            FakeDPO,
            DPOTrainerConfig().to_dict(),
            aliases={"max_prompt_length": "max_length"},
        )
        assert "max_prompt_length" not in out
        assert out["max_length"] == 2048

    def test_trackio_keeps_report_to_and_sets_project(self):
        from beni.core.srl.config import trl_config_kwargs

        class FakeGRPO:
            def __init__(self, report_to="trackio", project="huggingface", run_name=None):
                pass

        out = trl_config_kwargs(
            FakeGRPO, {"report_to": "trackio"}, project_name="sebeni-bam"
        )
        assert out["report_to"] == "trackio"
        assert out["project"] == "sebeni-bam"
        assert out["run_name"].startswith("run-")

    def test_trackio_project_dropped_when_unsupported(self):
        from beni.core.srl.config import trl_config_kwargs

        class FakeGRPO:
            def __init__(self, report_to="trackio"):
                pass

        out = trl_config_kwargs(
            FakeGRPO, {"report_to": "trackio"}, project_name="sebeni-bam"
        )
        assert out["report_to"] == "trackio"
        assert "project" not in out
        assert "run_name" not in out

    def test_installed_grpo_config_accepts_filtered_kwargs(self):
        trl = pytest.importorskip("trl")
        from beni.core.srl.config import GRPOTrainerConfig, trl_config_kwargs

        payload = trl_config_kwargs(
            trl.GRPOConfig,
            GRPOTrainerConfig(
                use_cpu=True,
                max_steps=1,
                report_to="none",
                output_dir="/tmp/sebeni-trl-kwargs",
            ).to_dict(),
        )
        args = trl.GRPOConfig(**payload)
        assert args.max_completion_length == 1024
        assert "max_prompt_length" not in payload or hasattr(args, "max_prompt_length")
        default = trl_config_kwargs(
            trl.GRPOConfig,
            GRPOTrainerConfig().to_dict(),
            project_name="sebeni-bam",
        )
        assert default.get("report_to") == "trackio"
        assert default.get("project") == "sebeni-bam"
        assert str(default.get("run_name") or "").startswith("run-")


class TestLogTrackio:
    def test_log_trackio_calls_trackio_log(self):
        from beni.core.srl.grpo import callbacks as cb

        fake = MagicMock()
        with patch.object(cb, "trackio", fake):
            cb.log_trackio({"loss": 1.5, "ok": True, "note": "x"}, step=3)
        fake.log.assert_called_once_with({"loss": 1.5}, step=3)

    def test_log_trackio_skips_when_uninstalled(self):
        from beni.core.srl.grpo import callbacks as cb

        with patch.object(cb, "trackio", None):
            cb.log_trackio({"loss": 1.0})

    def test_metrics_callback_logs_sebeni_reward(self):
        from beni.core.srl.grpo.callbacks import TrackioMetricsCallback

        rm = MagicMock()
        rm.total_reward = 0.8
        state = MagicMock()
        state.global_step = 2
        with patch("beni.core.srl.grpo.callbacks.log_trackio") as log_fn:
            TrackioMetricsCallback(reward_manager=rm).on_log(None, state, None)
        log_fn.assert_called_once_with({"sebeni/reward": 0.8}, step=2)


class TestRewardManager:
    """Test suite for RewardManager and custom reward registration."""

    def test_default_reward_functions(self):
        rm = RewardManager()
        funcs = rm.get_reward_functions()
        assert len(funcs) == 4
        func_names = [f.__name__ for f in funcs]
        assert "reward_format" in func_names
        assert "reward_morph" in func_names
        assert "reward_rule" in func_names
        assert "reward_lang" in func_names

    def test_disabled_reward_functions(self):
        cfg = RewardConfig(enable_rule_reward=False)
        rm = RewardManager(reward_config=cfg)
        funcs = rm.get_reward_functions()
        assert len(funcs) == 3
        func_names = [f.__name__ for f in funcs]
        assert "reward_rule" not in func_names

    def test_custom_reward_registration(self):
        rm = RewardManager()

        def custom_length_reward(completions: List[str], **kwargs) -> List[float]:
            return [float(len(c)) for c in completions]

        rm.register_custom_reward("length_reward", custom_length_reward, weight=0.5)
        funcs = rm.get_reward_functions()
        assert len(funcs) == 5

        # Test execution of custom reward wrapper
        last_func = funcs[-1]
        results = last_func(["hello", "world!!"])
        assert results == [2.5, 3.5]  # 5 * 0.5 and 7 * 0.5


class TestSebeniGrpoPipeline:
    """Test suite for SebeniGrpo pipeline functionality."""

    def test_pipeline_initialization(self):
        pipeline = SebeniGrpo()
        assert pipeline.config is not None
        assert pipeline.reward_manager is not None
        assert pipeline.hook_manager is not None
        assert pipeline.config.model.model_name == "HuggingFaceTB/SmolLM2-135M"

    def test_custom_config_initialization(self):
        master_cfg = MasterConfig(project_name="Pipeline-Test")
        pipeline = SebeniGrpo(config=master_cfg)
        assert pipeline.config.project_name == "Pipeline-Test"

    def test_register_reward_and_hook(self):
        pipeline = SebeniGrpo()

        def dummy_reward(completions, **kwargs):
            return [1.0] * len(completions)

        def dummy_hook(model, ref_model, batch_meta):
            pass

        pipeline.register_reward("dummy", dummy_reward, weight=0.2)
        pipeline.register_hook(dummy_hook)

        assert "dummy" in pipeline.reward_manager.custom_rewards
        assert len(pipeline.hook_manager.hooks) == 1

    def test_format_dataset(self):
        pipeline = SebeniGrpo()
        raw_sentences = [
            {
                "text": "aw ka ne labato.",
                "lang": "bm",
                "reference": {"tokens": [{"surface": "aw"}]}
            },
            {
                "text": "kàlanko jamanaw",
                "lang": "bm",
                "reference": {"tokens": [{"surface": "kàlanko"}]}
            }
        ]

        ds = pipeline.format_dataset(raw_sentences)

        assert len(ds) == 2
        assert "prompt" in ds.column_names
        assert "language" in ds.column_names
        assert "reference" in ds.column_names

        first_prompt = ds[0]["prompt"]
        assert first_prompt[0]["role"] == "system"
        assert first_prompt[1]["role"] == "user"
        assert first_prompt[1]["content"] == "aw ka ne labato."
        assert ds[0]["language"] == "bm"

    def test_run_batch_distillation_disabled(self):
        pipeline = SebeniGrpo()
        pipeline.config.distillation.enabled = False
        raw_sentences = [
            {"text": "test sentence", "lang": "bm"}
        ]
        results = pipeline.run_batch_distillation(raw_sentences)
        assert results == {}

    @patch("beni.core.srl.grpo.grpo.Distiller")
    def test_run_batch_distillation_enabled(self, mock_distiller_cls):
        from beni.core.morphotactic.distil.distillation import DistillProposal

        mock_distiller_instance = MagicMock()
        mock_distiller_instance.phi_on_texts.return_value = 0.2
        mock_distiller_instance.propose.return_value = DistillProposal(
            "g", "d", 0.2, 0.8, True, False
        )
        mock_distiller_instance.gram_path = "baseline_v2.gram"
        mock_distiller_instance.dict_path = "baseline_v2.dict"
        mock_distiller_cls.return_value = mock_distiller_instance

        pipeline = SebeniGrpo()
        pipeline.config.distillation.enabled = True
        pipeline.config.distillation.provider = "google"

        raw_sentences = [
            {"text": "aw ka ne labato.", "lang": "bm"},
            {"text": "kàlanko jamanaw", "lang": "bm"}
        ]

        results = pipeline.run_batch_distillation(raw_sentences)

        mock_distiller_cls.assert_called_once_with(
            lang_code="bam",
            backend="google",
            model="gemini-3.7-flash",
            working_dir=None,
            vertex=None,
            base_url=None,
            gguf_path=None,
            n_ctx=4096,
            max_input_chars=8000,
        )
        mock_distiller_instance.propose.assert_called_once_with(
            ["aw ka ne labato.", "kàlanko jamanaw"], current_phi=0.2
        )
        mock_distiller_instance.write_checkpoint.assert_called_once()
        assert "bam" in results
        assert results["bam"]["gram_path"] == "baseline_v2.gram"
        assert results["bam"]["allowed"] is True

    @patch("beni.core.srl.grpo.grpo.Distiller")
    def test_run_batch_distillation_mixed_languages(self, mock_distiller_cls):
        mock_distiller_instance = MagicMock()
        mock_distiller_instance.phi_on_texts.return_value = 0.9
        mock_distiller_cls.return_value = mock_distiller_instance

        pipeline = SebeniGrpo()
        pipeline.config.distillation.enabled = True
        results = pipeline.run_batch_distillation([
            {"text": "aw ka", "lang": "bam"},
            {"text": "i ni ce", "lang": "mku"},
        ])
        assert mock_distiller_cls.call_count == 2
        codes = {c.kwargs["lang_code"] for c in mock_distiller_cls.call_args_list}
        assert codes == {"bam", "mku"}
        assert set(results) == {"bam", "mku"}


class TestGRPOTrainingModel:
    """Test suite for model loading, PEFT adaptation, and training loop execution."""

    @pytest.mark.parametrize("model_name", TEST_MODELS)
    @patch("beni.core.srl.plugin.get_peft_model")
    @patch("beni.core.srl.plugin.AutoTokenizer.from_pretrained")
    @patch("beni.core.srl.plugin.AutoModelForCausalLM.from_pretrained")
    def test_load_models_and_lora(self, mock_model_from_pretrained, mock_tok_from_pretrained, mock_get_peft, model_name):
        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.chat_template = None

        mock_model_from_pretrained.return_value = mock_base_model
        mock_tok_from_pretrained.return_value = mock_tokenizer
        mock_get_peft.return_value = mock_peft_model

        pipeline = SebeniGrpo(model_name=model_name)
        pipeline.config.model.use_peft = True
        pipeline.config.model.load_in_4bit = True

        pipeline.load_models()

        mock_model_from_pretrained.assert_called_once()
        assert mock_tok_from_pretrained.call_args[0][0] == model_name
        mock_get_peft.assert_called_once_with(mock_base_model, ANY)
        assert pipeline.model == mock_peft_model
        assert pipeline.tokenizer == mock_tokenizer

    @patch("beni.core.srl.plugin.get_peft_model")
    @patch("beni.core.srl.plugin.AutoTokenizer.from_pretrained")
    @patch("beni.core.srl.plugin.AutoModelForCausalLM.from_pretrained")
    def test_load_models_skips_4bit_without_bitsandbytes(
        self, mock_model_from_pretrained, mock_tok_from_pretrained, mock_get_peft
    ):
        mock_tok_from_pretrained.return_value = MagicMock(
            pad_token=None, eos_token="<eos>", chat_template=None
        )
        mock_model_from_pretrained.return_value = MagicMock()
        mock_get_peft.return_value = MagicMock()
        pipeline = SebeniGrpo(model_name="HuggingFaceTB/SmolLM2-135M")
        pipeline.config.model.use_peft = True
        pipeline.config.model.load_in_4bit = True
        pipeline.config.trainer.use_cpu = False
        with patch("beni.core.srl.plugin.torch.cuda.is_available", return_value=False):
            pipeline.load_models()
        kwargs = mock_model_from_pretrained.call_args.kwargs
        assert kwargs.get("quantization_config") is None
        assert kwargs.get("device_map") == "cpu"

    @patch("beni.core.srl.plugin.get_peft_model")
    @patch("beni.core.srl.plugin.AutoTokenizer.from_pretrained")
    @patch("beni.core.srl.plugin.AutoModelForCausalLM.from_pretrained")
    def test_load_models_sets_chat_template_when_missing(
        self, mock_model_from_pretrained, mock_tok_from_pretrained, mock_get_peft
    ):
        from beni.core.srl.plugin import CHATML_CHAT_TEMPLATE

        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        tokenizer.chat_template = None
        mock_tok_from_pretrained.return_value = tokenizer
        mock_model_from_pretrained.return_value = MagicMock()
        mock_get_peft.return_value = MagicMock()
        pipeline = SebeniGrpo(model_name="HuggingFaceTB/SmolLM2-135M")
        pipeline.config.model.use_peft = False
        pipeline.config.model.load_in_4bit = False
        pipeline.config.trainer.use_cpu = True
        pipeline.load_models()
        assert pipeline.tokenizer.chat_template == CHATML_CHAT_TEMPLATE

    @patch("beni.core.srl.plugin.get_peft_model")
    @patch("beni.core.srl.plugin.AutoTokenizer.from_pretrained")
    @patch("beni.core.srl.plugin.AutoModelForCausalLM.from_pretrained")
    def test_load_models_keeps_existing_chat_template(
        self, mock_model_from_pretrained, mock_tok_from_pretrained, mock_get_peft
    ):
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "<eos>"
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        mock_tok_from_pretrained.return_value = tokenizer
        mock_model_from_pretrained.return_value = MagicMock()
        mock_get_peft.return_value = MagicMock()
        pipeline = SebeniGrpo(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
        pipeline.config.model.use_peft = False
        pipeline.config.model.load_in_4bit = False
        pipeline.config.trainer.use_cpu = True
        pipeline.load_models()
        assert pipeline.tokenizer.chat_template.startswith("{% for message in messages %}")
        assert "<|im_start|>" not in pipeline.tokenizer.chat_template

    @patch("beni.core.srl.plugin.write_model_card")
    @patch("beni.core.srl.grpo.grpo.GRPOConfig")
    @patch("beni.core.srl.grpo.grpo.GRPOTrainer")
    @patch("beni.core.srl.grpo.grpo.Distiller")
    def test_train_pipeline_execution(self, mock_distiller_cls, mock_trainer_cls, mock_grpo_config, mock_card, tmp_path):
        mock_distiller_inst = MagicMock()
        mock_distiller_cls.return_value = mock_distiller_inst

        mock_trainer_inst = MagicMock()
        mock_trainer_inst.train.return_value = {"train_loss": 0.123}
        mock_trainer_inst.optimizer = None
        mock_trainer_cls.return_value = mock_trainer_inst
        mock_grpo_config.return_value = MagicMock()

        pipeline = SebeniGrpo(model_name="HuggingFaceTB/SmolLM2-135M")
        pipeline.model = MagicMock()  # pre-loaded mock model
        pipeline.tokenizer = MagicMock()
        pipeline.config.distillation.enabled = True
        pipeline.config.trainer.use_cpu = True
        pipeline.config.trainer.output_dir = str(tmp_path)
        pipeline.config.trainer.push_to_hub = False

        raw_sentences = [
            {"text": "aw ka ne labato.", "lang": "bm", "reference": {"tokens": []}}
        ]

        result = pipeline.train(raw_sentences, project_name="Test-Run")

        mock_distiller_cls.assert_not_called()
        mock_trainer_cls.assert_called_once()
        callbacks = mock_trainer_cls.call_args.kwargs["callbacks"]
        assert all(type(cb).__name__ != "SelfAwareCallback" for cb in callbacks)
        mock_trainer_inst.train.assert_called_once()
        assert result == {"train_loss": 0.123}
