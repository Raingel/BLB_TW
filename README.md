# BLB_TW

This repository contains the processed dataset used in our study on how **weather data source heterogeneity** affects **rice bacterial leaf blight (BLB)** forecasting in Taiwan. It includes field-survey labels and survey-associated weather data (SAWD) extracted from multiple weather products.

## Repository structure

- `1_clean_data/`  
  Cleaned and labeled survey records used to build the event-based classification dataset (onset-related events and matched no-onset controls).  
  This folder also contains the metadata needed to link each labeled record to its corresponding SAWD.

- `2_weather_data/`  
  Survey-associated weather data (SAWD) extracted for each labeled survey record from each weather data source.  
  SAWD uses a fixed pre-survey window and is stored in per-sample files as used in the study.

- `4_packed_data/`  
  Packed model-input tables created from SAWD (e.g., wide-format features) and related preprocessing outputs used for model training and cross-source evaluation.

## Notes

- SAWD was constructed using the fixed pre-survey window and daily weather-variable set described in the accompanying manuscript.
- Weather inputs are provided as used in the analysis to support reproducibility and cross-source comparisons.

## Citation

If you use this dataset, please cite this repository directly.

The associated manuscript is currently under review. We will update this section with the formal paper citation after publication.

## License

Data in this repository are released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

## Contact

For questions or issues related to this dataset, please open a GitHub issue in this repository.
