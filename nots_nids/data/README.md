# Dataset Download Instructions

## CICIDS-2017 (PRIMARY — use this first)

**URL:** <https://www.unb.ca/cic/datasets/ids-2017.html>

Download **"MachineLearningCSV.zip"** (pre-extracted features, ~500 MB).
Do **NOT** download the raw PCAPs (50 GB).

```bash
mkdir -p data/cicids2017
# Extract the zip into data/cicids2017/
unzip MachineLearningCSV.zip -d data/cicids2017/
```

Expected files after extraction:
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`

---

## UNSW-NB15 (SECONDARY)

**URL:** <https://research.unsw.edu.au/projects/unsw-nb15-dataset>

Files needed:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`
- `NUSW-NB15_features.csv` (column descriptions)

```bash
mkdir -p data/unsw_nb15
# Place files in data/unsw_nb15/
```

---

## NSL-KDD (COMPARISON BASELINE ONLY)

**URL:** <https://www.unb.ca/cic/datasets/nsl.html>

Files needed:
- `KDDTrain+.txt`
- `KDDTest+.txt`

```bash
mkdir -p data/nsl_kdd
# Place files in data/nsl_kdd/
```
