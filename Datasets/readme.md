## Dataset Organization and Download Links

Two files are downloaded; everything else in this tree is generated locally by
`DataGeneration/`. Only the two originals are required to reproduce the study
from scratch.

### Original data sources (download these)

- **UAV-ATTACK Dataset** — binary, benign vs malicious  
  Download from: [IEEE DataPort](https://ieee-dataport.org/open-access/uav-attack-dataset) (needs an IEEE account)  
  Place `spoofing-merged-gps-only.csv` in:
  ```
  Datasets/csv/UAVATTACK/
  ```
  3,622 rows x 84 columns. The authors recommend the **live** spoofing/jamming
  captures over the simulated ones.

- **2D Spoofing Dataset** — four classes, benign / simple / intermediate / sophisticated  
  Download from: [Mendeley Data](https://data.mendeley.com/datasets/z7dj3yyzt8/3)  
  Place `GPS_Data_Simplified_2D_Feature_Map.xlsx` in:
  ```
  Datasets/excel/2dSpoofingDataset/
  ```
  510,530 rows x 14 columns.

### Generated beside the originals

| File pattern | Produced by | Datasets |
|---|---|---|
| `spoofing_{gc,ctgan,tvae}_{10000,100000,1000000}.csv` | `DataGeneration/SyntheticData/gen_synth.py`, or `SDV_Synthetic.ipynb` while fitting, sampling the SDV synthesizers | both |
| `syntheticSpoofing_gan_data.csv` | `DataGeneration/SyntheticData/<dataset>/GAN/GAN.ipynb` | both |
| `syntheticSpoofing_cgan_data.csv` | `DataGeneration/SyntheticData/<dataset>/CGAN/CGAN.ipynb` | both |

Every generated table is written through the canonical dataset schema
(`gps_ids/schema.py`), so it carries the same feature columns, in the same
order, with integer labels — whichever generator produced it.

Both datasets have all five generators, so each one's synthetic model runs cover
eleven files: three SDV synthesizers at three sample sizes, plus GAN and CGAN.
The GAN and CGAN models for 2D must be generated locally — only the UAV ones
ship in the repository.

Pre-generated copies of our synthetic models and of the binary images:
[Google Drive](https://drive.google.com/drive/folders/1l2LfGwJ8FRzE115wkMUHQRaGiru-SwGv)

The binary images are **not** stored here. They live next to the code that
renders them, under `DataGeneration/BinaryImages/<dataset>/image-results-spoofing/`.

Both dataset folders are gitignored, so nothing in this tree is committed apart
from the two UAV GAN/CGAN tables and this readme.

### Final folder structure

```plaintext
Datasets/
├── csv/
│   └── UAVATTACK/
│       ├── spoofing-merged-gps-only.csv          <- downloaded
│       ├── spoofing_{gc,ctgan,tvae}_10000.csv    <- generated
│       ├── spoofing_{gc,ctgan,tvae}_100000.csv   <- generated
│       ├── spoofing_{gc,ctgan,tvae}_1000000.csv  <- generated
│       ├── syntheticSpoofing_gan_data.csv
│       └── syntheticSpoofing_cgan_data.csv
└── excel/
    └── 2dSpoofingDataset/
        ├── GPS_Data_Simplified_2D_Feature_Map.xlsx   <- downloaded
        ├── spoofing_{gc,ctgan,tvae}_10000.csv        <- generated
        ├── spoofing_{gc,ctgan,tvae}_100000.csv       <- generated
        ├── spoofing_{gc,ctgan,tvae}_1000000.csv      <- generated
        ├── syntheticSpoofing_gan_data.csv                 <- generated
        └── syntheticSpoofing_cgan_data.csv                <- generated
```

Roughly 5.5 GB of generated CSVs at the full three sample sizes.
