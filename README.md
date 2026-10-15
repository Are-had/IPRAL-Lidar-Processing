# IPRAL-Lidar-Processing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23387966.svg)](https://doi.org/10.5281/zenodo.23387966)

Research code developed at LMD/IPSL to calibrate the **IPRAL 355 nm far-range lidar** signals (SIRTA, Palaiseau) and retrieve particle backscatter, **for the purpose of comparing them with ATLID/EarthCARE**. It implements the algorithms described in the technical note:

> Haddouche, A. and Feofilov, A. G. (2026). *Calibration Procedure and Backscatter Retrieval for IPRAL 355 nm Far-Range Signals.* Technical Note, v1.0, LMD/IPSL.

and was used to produce the results of the ATLID/EarthCARE zenith-to-nadir comparison study (Feofilov, Haddouche et al., in preparation).

## Disclaimer

**This is not the official IPRAL processing.** It is a research processing chain written for one specific application, the ATLID/IPRAL comparison, and is shared to make that study reproducible.

- **Not operational IPRAL processing.** It does not replace or represent the official IPRAL/SIRTA data-processing procedures.
- **No formal ACTRIS compliance claimed.** The procedure has not been submitted to or endorsed by ACTRIS or another network.
- **Not an instrument characterization.** The authors did not have access to the instrument's optical setup. The calibration and cross-talk coefficients are effective empirical parameters of the combined optical and detection response, not measurements of individual optical components.
- **Application-specific polarization correction.** It does not claim equivalence to dedicated polarization-calibration methods (e.g. Freudenthaler et al., 2009).
- **Validation limited to the stated application.** Suitability for aerosol climatology, independent depolarization products or other network-level applications has not been established.

The processing chain may be revised in light of future independent instrument-level calibration and characterization work.

---

## Processing chain

| Step | Technical note section | Code |
|---|---|---|
| Read L0 netCDF, background subtraction, range correction (RCS) | Sect. 2 | `src/io_tools.py` — `read_nc_file_`, `substract_bckgrd`, `get_corrected_signal` |
| Molecular profile from radiosonde (β_mol, α_mol, AMB_clear) | Sect. 3.1 | `src/l1_tools.py` — `conc_calculation`, `calculate_AMB_clear` |
| Calibration coefficient *d* and residual background Δ (analytical init + Nelder-Mead on MAPE) | Sect. 3.3.1 | `src/l1_tools.py` — `Calibration`, `cost_function`, `optimize` |
| Analog / photon-counting merging (Hanning cross-fade) | Sect. 3.3.2 | `src/l1_tools.py` — `merged_signal_hanning` <br> `notebook/Merge_signals.ipynb` |
| Case 1 — negligible cross-talk | Sect. 3.3 | `notebook/01_L1_Calibration_Case1_NoCrossTalk.ipynb` |
| Case 2 — cross-talk with reference depolarizations (molecular + cirrus) | Sect. 3.4 | `notebook/02_L1_Calibration_Case2_KnownDepol.ipynb` |
| Case 3 — cross-talk with reference ATB (constant *A*) | Sect. 3.5 | `notebook/03_L1_Calibration_Case3_KnownATB.ipynb` |
| Klett-Fernald-Sasano inversion, per-channel inversion and negative-value filtering | Sect. 4 | `src/l2_tools.py` — `klett_backscatter_aerosol_simplifié`, `L1_2_L2`, `filter_negative_backscatter` <br> `notebook/04_L2_Aerosol_Inversion_Klett.ipynb` |

## Repository structure

```
IPRAL-Lidar-Processing/
├── src/
│   ├── io_tools.py        # L0 reading, background subtraction, RCS, quicklooks
│   ├── l1_tools.py        # molecular profile, calibration (d, Δ), signal merging
│   └── l2_tools.py        # Klett inversion and negative-backscatter filtering
├── notebook/
│   ├── data_preprocessing/
│   │   ├── procees_data_Case_1_and_3.ipynb   # L0 + radiosonde → mean profiles (2025-07-04)
│   │   └── procees_data_Case_2.ipynb         # L0 + radiosonde → mean profiles (2025-06-11)
│   ├── 01_L1_Calibration_Case1_NoCrossTalk.ipynb
│   ├── 02_L1_Calibration_Case2_KnownDepol.ipynb
│   ├── 03_L1_Calibration_Case3_KnownATB.ipynb
│   ├── Merge_signals.ipynb
│   └── 04_L2_Aerosol_Inversion_Klett.ipynb
├── requirements.txt
├── CITATION.cff
├── .zenodo.json
└── LICENSE
```

## Data

The data are not included in this repository because of their size (about 2 GB). They are available on request from the authors: Arezki Haddouche (arezki.haddouche@lmd.ipsl.fr) or Artem Feofilov (afeofilo@lmd.ipsl.fr). The IPRAL L0 files can also be obtained directly from the SIRTA data portal (see below).

To run the notebooks, place the data in a `data/` folder at the root of the repository:
```
data/
├── Case_1_and_3/            # 2025-07-04
│   ├── raw/                 # IPRAL L0 + radiosonde
│   └── processed/           # clear/cloudy-sky mean profiles, molecular reference, L1 ATB, calibration constants
└── Case_2/                  # 2025-06-11
    ├── raw/                 # IPRAL L0 + radiosonde
    └── processed/           # clear/cloudy-sky mean profiles, molecular reference
```

- **IPRAL L0 files** (`ipral_1a_Lz1Lb87R15mF30sPbckIqsmart_v01_YYYYMMDD_000000_1440.nc`): 355 nm far-range channels `rcs_02`, `rcs_03` (parallel, analog / photon-counting) and `rcs_04`, `rcs_05` (cross, analog / photon-counting), 15 m vertical and 30 s temporal resolution. Original source: [SIRTA data portal](https://sirta.ipsl.polytechnique.fr/).
- **Radiosonde profiles** (`rs_2a_GRUAN_TRP_M10_YYYYMMDD_*.nc`): GRUAN M10 soundings from Trappes (Météo-France 07145).

## Usage

```bash
git clone https://github.com/Are-had/IPRAL-Lidar-Processing.git
cd IPRAL-Lidar-Processing
pip install -r requirements.txt
```

Then run the notebooks in this order:

1. `notebook/data_preprocessing/` — builds the mean clear-sky / cloudy-sky profiles and the molecular reference (requires the IPRAL L0 files; its outputs are also provided in `data/<case>/processed/`, so this step can be skipped).
2. `01_L1_Calibration_Case1_NoCrossTalk.ipynb`, `02_L1_Calibration_Case2_KnownDepol.ipynb`, `03_L1_Calibration_Case3_KnownATB.ipynb` — L1 calibration for each case.
3. `Merge_signals.ipynb` — analog / photon-counting merging and total ATB.
4. `04_L2_Aerosol_Inversion_Klett.ipynb` — L2 particle backscatter.

## Requirements

Python ≥ 3.9 with `numpy`, `scipy`, `pandas`, `matplotlib`, `netCDF4` (see `requirements.txt`).

## How to cite

If you use this code, please cite the archived version on Zenodo:

> Haddouche, A. and Feofilov, A. G. (2026). *IPRAL-Lidar-Processing: calibration and backscatter retrieval for IPRAL 355 nm far-range signals* (Version 1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23387966

and the technical note (v1.0) describing the algorithms.

## Authors

- **Arezki Haddouche** — LMD/IPSL, École Polytechnique — arezki.haddouche@lmd.ipsl.fr
- **Artem Feofilov** — LMD/IPSL, École Polytechnique — afeofilo@lmd.ipsl.fr

## Acknowledgements

The apprenticeship of A. Haddouche was funded by the Académie Spatiale d'Île-de-France, a "Compétences et Métiers d'Avenir" project of France 2030 managed by the French National Research Agency (ANR) under reference ANR-23-CMAS-0041. IPRAL data are provided by the SIRTA observatory (IPSL).

## License

Released under the MIT License — see [`LICENSE`](LICENSE).