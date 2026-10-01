"""Anchored Preference Optimization (APO, https://huggingface.co/papers/2408.06266) policy-update plugin — TRL DPO with ``apo_zero`` / ``apo_down`` loss."""

from __future__ import annotations

from beni.core.srl.dpo.dpo import SebeniDpo


class SebeniApo(SebeniDpo):
    """Anchored Preference Optimization (APO) as a TRL DPO ``loss_type`` plugin on the same SAMPG loop."""

    name = "apo"

    def train(self, data, project_name=None, run_distillation_first=False, extra_callbacks=None):
        from dataclasses import fields

        for f in fields(self.config.apo):
            if hasattr(self.config.dpo, f.name):
                setattr(self.config.dpo, f.name, getattr(self.config.apo, f.name))
        print("Starting Sebeni APO Training Loop (SAMPG)...")
        return super().train(
            data,
            project_name=project_name,
            run_distillation_first=run_distillation_first,
            extra_callbacks=extra_callbacks,
        )
