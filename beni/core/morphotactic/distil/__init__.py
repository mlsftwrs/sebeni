import json

from dataclasses import dataclass
from pydantic import BaseModel, Field

class DistilOutput(BaseModel):
    sebeni_gram: str = Field(description="The updated .gram content.")
    sebeni_dict: str = Field(description="The updated .dict content.")

@dataclass
class DistilSysPrompt:
    language: str
    samples: dict  # Sample File: Content
    current_gram: str = None
    current_dict: str = None
    mode: str = "delta"  # delta | bootstrap

    def prompt(self) -> str:
        if self.mode == "bootstrap":
            return self._bootstrap_prompt()
        return self._delta_prompt()

    def _bootstrap_prompt(self) -> str:
        return f"""
        You are an expert linguist and computational morphologist specializing in the SebenX / Daba morphological parser family (used for Mande and related West-African languages such as Bamana/Bambara, Maninka, Dogon, etc.).

        Language: {self.language}

        There is **no existing grammar or dictionary** for this language. You must emit an initial Daba-compatible `.gram` and `.dict` from:
        1. the language metadata above,
        2. the format samples below,
        3. the input texts in the user message.

        ### Sample files that define the required output formats
        Use these exact formats, conventions, macros, section names, pattern syntax, and dictionary field structure. Do not invent new formalisms.

        {json.dumps(self.samples, indent=2)}

        ### Output
        Return valid JSON with **full file contents** (not delta tags):
        {{
            "sebeni_gram": "<complete .gram file>",
            "sebeni_dict": "<complete .dict file in MDF>"
        }}
        The grammar must include a `plan` / `for token:` section so Daba's GrammarLoader can parse it.
    """

    def _delta_prompt(self) -> str:
        return rf"""
        You are an expert linguist and computational morphologist specializing in the SebenX / Daba morphological parser family (used for Mande and related West-African languages such as Bamana/Bambara, Maninka, Dogon, etc.).

        Language: {self.language}

        ### Sample files that define the required output formats
        Use these exact formats, conventions, macros, section names, pattern syntax, and dictionary field structure as your reference. Do not invent new formalisms.

        {json.dumps(self.samples, indent=2)}

        ### BASELINE Checkpoints (Cached Reference)
        BASE GRAMMAR:
        {self.current_gram}

        BASE DICTIONARY:
        {self.current_dict}

        ### Conflict Handling & Delta Instructions
        1. Analyze the new text against the BASELINE checkpoints and any PREVIOUS DELTAS.
        2. Identify conflicts, errors, or changes required by the new text:
        - **New Material:** Mark with `[ADD]`
        - **Correction/Modification:** Mark with `[REPLACE: target_identifier]` where `target_identifier` is the exact `\\lx` entry or rule name being corrected.
        - **Deprecation/Deletion:** Mark with `[DELETE: target_identifier]` or `[REMOVE: target_identifier]`
        3. Formatting Rules for Deltas:
        - For Dictionary modifications, specify the target lexeme:
            `[REPLACE: \\lx lexeme_name]` followed by the full updated MDF entry block.
        - For Grammar modifications, specify the macro or section name:
            `[REPLACE: @macro_name]` or `[REPLACE: rule_pattern]` followed by the updated rule.
        4. Output must be valid JSON returning ONLY the delta actions:
        {{
            "sebeni_gram": "<action-tagged grammar operations>",
            "sebeni_dict": "<action-tagged dictionary operations>"
        }}
    """

@dataclass
class DistilUserPrompt:
    text: str
    gram_deltas: str = None
    dict_deltas: str = None

    def prompt(self) -> str:
        return f"""
        ### Previously Generated Deltas (Updates added in prior runs)
        GRAMMAR DELTAS SO FAR:
        {self.gram_deltas if self.gram_deltas else "[NONE]"}

        DICTIONARY DELTAS SO FAR:
        {self.dict_deltas if self.dict_deltas else "[NONE]"}

        ### New Text to Analyse
        ---
        {self.text}
        ---

        Analyse the new text using the cached Baseline AND the Deltas provided above. Return ONLY the new or updated entries required for this text chunk in the requested JSON structure.
    """

