# Data Card

- **Dataset name:** Preference Alignment Lab sample preferences
- **Source:** Repository-provided `data/sample_preferences.jsonl`; upstream provenance is not documented.
- **License/permission:** Not specified. Confirm usage rights before redistributing or extending the data.
- **Schema:** JSONL records with non-empty `prompt`, `chosen`, and `rejected` strings plus optional `metadata`. Chosen and rejected responses must be meaningfully different.
- **Labeling rubric:** Technical accuracy and helpfulness, primarily for machine-learning education questions.
- **Known biases:** All 24 records are English educational prompts. The preferred responses are generally longer and more detailed, so length can become a shortcut for preference.
- **Safety/PII checks:** Optional email/phone guardrails pass on the current sample. These regex checks are not a substitute for a complete privacy or safety review.
- **Train/validation/test split method:** Deterministic prompt-grouped split with seed 42. At the configured 20% validation ratio, the current data produces 19 training and 5 validation examples, with no prompt overlap. No test split is defined.
