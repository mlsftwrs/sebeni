# `beni.core.safety`

Gates on promote, policy update (SFT, GRPO, DPO, and APO), and Hub push. See
[Safety](../safety.md) for each gate.

```python
from beni.core.safety import SafetyGovernor, SafetySpec

gov = SafetyGovernor(SafetySpec(tau=0.5))
d = gov.allow_promote(phi=0.4, phi_prime=0.6, parseable=True, first_create=False)
assert d.allowed
```

::: beni.core.safety.spec.SafetySpec
    options:
      members: []
      filters:
        - "!^_"
        - "!^fired$"
        - "!^reset_fired$"
      show_root_heading: true
      heading_level: 2
      merge_init_into_class: true
      show_if_no_docstring: false

::: beni.core.safety.governor.SafetyGovernor
    options:
      members:
        - allow_promote
        - allow_policy_update
        - allow_hub_push
        - record_snapshot
        - write_snapshot
        - apply_pre_update
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: true

::: beni.core.safety.governor.PromoteDecision
    options:
      members: []
      show_root_heading: true
      heading_level: 2
      merge_init_into_class: true
      show_if_no_docstring: false

::: beni.core.safety.governor.SafetySnapshot
    options:
      members:
        - to_dict
        - write
        - load
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: true
