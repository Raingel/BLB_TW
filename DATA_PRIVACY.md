# Data privacy

The original surveillance records include information that could identify participating farms. To protect those farms, the public training tables exclude:

- Original field names and survey notes
- Survey dates and coordinates
- Weather-station identifiers
- Exact reanalysis grid coordinates
- Contact information

The public `record_id`, `pair_id`, and `FieldID` values are newly assigned anonymous identifiers. The excluded fields were not model inputs and are not required to reproduce the reported model training or cross-source evaluation.

The repository does not contain API keys, authentication credentials, private cache files, or local user paths.

