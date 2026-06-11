# Public training data

The two CSV files contain the same 404 labeled records and privacy-safe metadata.

## Sequence-format table

`blb_training_sawd_sequences_public.csv` is the compact model-ready form. It contains 50 SAWD columns: five weather sources x ten variables. Each SAWD cell is a serialized list of 19 daily values ordered from day -21 through day -3 relative to the private survey date.

## Flattened wide table

`blb_training_sawd_wide_public.csv` contains one numeric column for every source-variable-day combination. It has four public metadata columns and 950 SAWD columns.

## Anonymous grouping fields

- `record_id`: unique public record identifier.
- `pair_id`: links the class-0 and class-1 records in one event-control pair.
- `FieldID`: anonymous field grouping used for grouped cross-validation. A field may contain more than one pair.

## Privacy treatment

Original field names, survey dates, coordinates, survey notes, station identifiers, exact reanalysis grid coordinates, and contact information are excluded. These fields were not model inputs and are not needed to reproduce model training or cross-source evaluation.
