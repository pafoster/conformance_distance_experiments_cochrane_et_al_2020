# Anomaly Detection on Streamed Data - Code for Numerical Experiments
This code implements the numerical experiments described in the paper manuscript "Anomaly Detection on Streamed Data".

Experiments are implemented in Python as a set of Jupyter notebooks, with each notebook corresponding to a section in the paper and responsible for generating the experimental results reported in that section.

Before executing the notebooks, please ensure that the requirements listed in the section below are met. Executing the notebooks should then generate the following results:

## [Section 4.1 (Handwritten digits): pen_digit_anomalies/pen_digit_anomalies.ipynb](pen_digit_anomalies/pen_digit_anomalies.ipynb)

Table 1: 

| Signature order 1 | Signature order 2 | Signature order 3 | Signature order 4 | Signature order 5 |
|-------------------|-------------------|-------------------|-------------------|-------------------|
| 0.901 +- 0.004    | 0.965 +- 0.002    | 0.983 +- 0.001    | 0.987 +- 0.001    | 0.989 +- 0.000    |

Results generated by executing notebook up to code cells #14 and #15.

Figure 3: Image files located at [pen_digit_anomalies/ecdf_order_1.pdf](pen_digit_anomalies/ecdf_order_1.pdf), [pen_digit_anomalies/ecdf_order_2.pdf](pen_digit_anomalies/ecdf_order_2.pdf), [pen_digit_anomalies/ecdf_order_3.pdf](pen_digit_anomalies/ecdf_order_3.pdf), [pen_digit_anomalies/ecdf_order_4.pdf](pen_digit_anomalies/ecdf_order_4.pdf), [pen_digit_anomalies/ecdf_order_5.pdf](pen_digit_anomalies/ecdf_order_5.pdf) generated by executing notebook up to cell #16.

## [Section 4.2 (Marine vessel traffic data): ship_movement_anomalies/ship_movement_anomalies.ipynb](ship_movement_anomalies/ship_movement_anomalies.ipynb)

Figure 1: Image files located at [ship_movement_anomalies/paths_short_vessel.pdf](ship_movement_anomalies/paths_short_vessel.pdf) and [ship_movement_anomalies/paths_long_vessel.pdf](ship_movement_anomalies/paths_long_vessel.pdf) generated by executing notebook up to code cell #22.

Table 2:

|                                              | Substreams using distance threshold 4000m | Substreams using distance threshold 8000m | Substreams using distance threshold 16000m | Substreams using distance threshold 32000m |
|----------------------------------------------|-------------------------------------------|-------------------------------------------|--------------------------------------------|--------------------------------------------|
| Lead/Lag=False, Time=False, Inv. Reset=False | 0.723 +- 0.005                            | 0.706 +- 0.005                            | 0.705 +- 0.005                             | 0.740 +- 0.005                             |
| Lead/Lag=False, Time=False, Inv. Reset=True  | 0.776 +- 0.005                            | 0.789 +- 0.004                            | 0.785 +- 0.004                             | 0.805 +- 0.004                             |
| Lead/Lag=False, Time=True, Inv. Reset=False  | 0.810 +- 0.004                            | 0.813 +- 0.004                            | 0.818 +- 0.004                             | 0.848 +- 0.004                             |
| Lead/Lag=False, Time=True, Inv. Reset=True   | 0.839 +- 0.004                            | 0.860 +- 0.004                            | 0.863 +- 0.004                             | 0.879 +- 0.004                             |
| Lead/Lag=True, Time=False, Inv. Reset=False  | 0.811 +- 0.004                            | 0.835 +- 0.004                            | 0.824 +- 0.004                             | 0.837 +- 0.004                             |
| Lead/Lag=True, Time=False, Inv. Reset=True   | 0.812 +- 0.004                            | 0.835 +- 0.004                            | 0.833 +- 0.004                             | 0.855 +- 0.004                             |
| Lead/Lag=True, Time=True, Inv. Reset=False   | 0.845 +- 0.004                            | 0.861 +- 0.004                            | 0.862 +- 0.004                             | 0.877 +- 0.003                             |
| Lead/Lag=True, Time=True, Inv. Reset=True    | 0.848 +- 0.004                            | 0.863 +- 0.004                            | 0.870 +- 0.003                             | 0.891 +- 0.003                             |

Results generated by executing notebook up to code cells #42 and #43.

Table 2: 

|                                              | Substreams using distance threshold 4000m | Substreams using distance threshold 8000m | Substreams using distance threshold 16000m | Substreams using distance threshold 32000m |
|----------------------------------------------|-------------------------------------------|-------------------------------------------|--------------------------------------------|--------------------------------------------|
| Lead/Lag=False, Time=False, Inv. Reset=False | 0.690 +- 0.005                            | 0.718 +- 0.005                            | 0.717 +- 0.005                             | 0.733 +- 0.005                             |
| Lead/Lag=False, Time=False, Inv. Reset=True  | 0.682 +- 0.005                            | 0.698 +- 0.005                            | 0.714 +- 0.005                             | 0.716 +- 0.005                             |
| Lead/Lag=False, Time=True, Inv. Reset=False  | 0.771 +- 0.005                            | 0.779 +- 0.005                            | 0.779 +- 0.005                             | 0.803 +- 0.004                             |
| Lead/Lag=False, Time=True, Inv. Reset=True   | 0.745 +- 0.005                            | 0.751 +- 0.005                            | 0.761 +- 0.005                             | 0.797 +- 0.004                             |
| Lead/Lag=True, Time=False, Inv. Reset=False  | 0.759 +- 0.005                            | 0.765 +- 0.005                            | 0.766 +- 0.005                             | 0.763 +- 0.005                             |
| Lead/Lag=True, Time=False, Inv. Reset=True   | 0.755 +- 0.005                            | 0.761 +- 0.005                            | 0.763 +- 0.005                             | 0.762 +- 0.005                             |
| Lead/Lag=True, Time=True, Inv. Reset=False   | 0.820 +- 0.004                            | 0.815 +- 0.004                            | 0.823 +- 0.004                             | 0.817 +- 0.004                             |
| Lead/Lag=True, Time=True, Inv. Reset=True    | 0.810 +- 0.004                            | 0.795 +- 0.005                            | 0.816 +- 0.004                             | 0.815 +- 0.004                             |

Results generated by executing notebook up to code cells #47 and #48.

## [Section 4.3 (Univariate time series): ucr_comparisons/ucr_dataset_comparison.ipynb](ucr_comparisons/ucr_dataset_comparison.ipynb)

Figure 2: Image file located at [ucr_comparisons/benchmark.pdf](ucr_comparisons/benchmark.pdf) generated by executing notebook up to code cell #12.

Table 3:

| Dataset               | Conformance (0.1% anomaly rate) | ADSL (0.1% anomaly rate) | Conformance (5% anomaly rate) | ADSL (5% anomaly rate) |
|-----------------------|---------------------------------|--------------------------|-------------------------------|------------------------|
| Adiac                 | 1.00 (0.00)                     | 0.99 (0.10)              | 0.99 (0.09)                   | 0.95 (0.05)            |
| ArrowHead             | 0.80 (0.07)                     | 0.65 (0.03)              | 0.74 (0.06)                   | 0.64 (0.03)            |
| Beef                  | 0.80 (0.22)                     | 0.57 (0.15)              | 0.80 (0.22)                   | 0.73 (0.12)            |
| BeetleFly             | 0.75 (0.08)                     | 0.90 (0.08)              | 0.69 (0.06)                   | 0.84 (0.08)            |
| BirdChicken           | 0.75 (0.13)                     | 0.85 (0.15)              | 0.77 (0.15)                   | 0.79 (0.09)            |
| CBF                   | 0.97 (0.01)                     | 0.80 (0.04)              | 0.86 (0.03)                   | 0.68 (0.03)            |
| ChlorineConcentration | 0.91 (0.01)                     | 0.50 (0.00)              | 0.88 (0.01)                   | 0.47 (0.01)            |
| Coffee                | 0.80 (0.05)                     | 0.84 (0.04)              | 0.78 (0.05)                   | 0.73 (0.05)            |
| ECG200                | 0.80 (0.06)                     | 0.50 (0.03)              | 0.75 (0.05)                   | 0.47 (0.04)            |
| ECGFiveDays           | 0.97 (0.02)                     | 0.94 (0.11)              | 0.83 (0.02)                   | 0.86 (0.01)            |
| FaceFour              | 0.78 (0.10)                     | 0.94 (0.10)              | 0.78 (0.13)                   | 0.88 (0.11)            |
| GunPoint              | 0.85 (0.05)                     | 0.75 (0.03)              | 0.81 (0.05)                   | 0.68 (0.04)            |
| Ham                   | 0.52 (0.04)                     | 0.50 (0.02)              | 0.52 (0.04)                   | 0.50 (0.03)            |
| Herring               | 0.58 (0.06)                     | 0.52 (0.02)              | 0.56 (0.05)                   | 0.49 (0.04)            |
| Lightning2            | 0.73 (0.04)                     | 0.63 (0.07)              | 0.75 (0.05)                   | 0.50 (0.07)            |
| Lightning7            | 0.94 (0.09)                     | 0.73 (0.11)              | 0.82 (0.09)                   | 0.68 (0.07)            |
| Meat                  | 0.94 (0.03)                     | 1.00 (0.04)              | 0.79 (0.07)                   | 0.87 (0.05)            |
| MedicalImages         | 0.97 (0.03)                     | 0.90 (0.03)              | 0.95 (0.04)                   | 0.83 (0.05)            |
| MoteStrain            | 0.89 (0.01)                     | 0.74 (0.01)              | 0.86 (0.02)                   | 0.71 (0.03)            |
| Plane                 | 1.00 (0.00)                     | 1.00 (0.04)              | 1.00 (0.04)                   | 1.00 (0.04)            |
| Strawberry            | 0.92 (0.01)                     | 0.77 (0.03)              | 0.88 (0.01)                   | 0.67 (0.02)            |
| Symbols               | 1.00 (0.01)                     | 0.96 (0.02)              | 0.99 (0.01)                   |  0.95 (0.03)           |
| ToeSegmentation1      | 0.77 (0.03)                     | 0.95 (0.01)              | 0.76 (0.05)                   | 0.84 (0.03)            |
| ToeSegmentation2      | 0.80 (0.06)                     | 0.88 (0.02)              | 0.77 (0.06)                   | 0.80 (0.10)            |
| Trace                 | 1.00 (0.00)                     | 1.00 (0.04)              | 1.00 (0.05)                   | 1.00 (0.02)            |
| TwoLeadECG            | 0.92 (0.02)                     | 0.89 (0.01)              | 0.82 (0.02)                   | 0.81 (0.02)            |
| Wafer                 | 0.97 (0.02)                     | 0.56 (0.02)              | 0.81 (0.03)                   | 0.53 (0.01)            |
| Wine                  | 0.85 (0.06)                     | 0.53 (0.02)              | 0.81 (0.09)                   | 0.53 (0.02)            |


Results generated by executing notebook up to code cell #12.

# Requirements

The notebooks were implemented using Python 3.7. The list of Python package dependencies is defined in [requirements.txt](requirements.txt). A typical process for installing the package dependencies involves creating a new Python virtual environment and then inside the environment executing

    pip install -r requirements.txt

As listed in the aforementioned requirements file, the relevant dependency for computing path signatures is the package [iisignature](https://pypi.org/project/iisignature/). For notes on operating systems supported by iisignature and how to resolve installation issues, please refer to the project's [documentation](https://github.com/bottler/iisignature)

When executed, the notebooks attempt to download and extract required datasets automatically. For this purpose, the following command-line utilities must be available in the user's environment:

    unzip
    uncompress

Both utilities are typically present on MacOS and Linux installations. For the case when it is not possible to execute these aforementioned utilities successfully from within the notebook, we advise the user to download and extract the datasets manually, before executing the notebooks.

# Testing Notes

The code has been verified to execute successfully under MacOS Catalina 10.15.3 and Windows 10.

# Execution Times

The following execution times are based on running the notebooks on a 2018 MacBook Pro equipped with a 2.6 GHz 6-Core Intel Core i7 processor and 32 GB 2400 MHz DDR4 memory. We report separately the time observed for computing path signature variances alone, and the time observed for executing the entire notebook. We measured the former using the %%time command and the latter by using Python's time module.

## [Section 4.1 (Handwritten digits): pen_digit_anomalies/pen_digit_anomalies.ipynb](pen_digit_anomalies/pen_digit_anomalies.ipynb)

Code cell #11, corresponding to results in Table 1: Wall time 0:15h, Total CPU Time 0:54h

Entire notebook: Wall time 0:18h

## [Section 4.2 (Marine vessel traffic data): ship_movement_anomalies/ship_movement_anomalies.ipynb](ship_movement_anomalies/ship_movement_anomalies.ipynb)

Code cell #38, corresponding to results in Table 2: Wall time 9:26h, Total CPU Time 2d 3:51h

Entire notebook: Wall time 11:42h

## [Section 4.3 (Univariate time series): ucr_comparisons/ucr_dataset_comparison.ipynb](ucr_comparisons/ucr_dataset_comparison.ipynb)

Code cell #9, corresponding to results in Figure 2: Wall time 1:44h, Total CPU Time 4:59h

Entire notebook: Wall time 1:44h