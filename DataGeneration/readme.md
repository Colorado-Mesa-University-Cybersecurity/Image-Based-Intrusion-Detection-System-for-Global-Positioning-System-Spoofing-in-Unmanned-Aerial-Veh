## Data Generation

Everything derived from the two original datasets is produced here. Nothing in
this tree trains a detector; the models that consume these outputs live in
`Models/<dataset>/`.

```plaintext
DataGeneration/
├── BinaryImages/
│   ├── 2D/
│   │   ├── tabToImage.ipynb          renders every row to a binary image
│   │   ├── split_images.py           carves the test split, idempotently
│   │   └── image-results-spoofing/   generated images and checkpoints, gitignored
│   └── UAVATTACK/
│       ├── tabToImage.ipynb
│       ├── split_images.py
│       └── image-results-spoofing/
└── SyntheticData/
    ├── gen_synth.py                  samples the fitted synthesizers
    ├── 2D/
    │   ├── SDV/                      fit notebook + metadata; the 3 .pkl are not committed
    │   ├── GAN/GAN.ipynb
    │   └── CGAN/CGAN.ipynb
    └── UAVATTACK/
        ├── SDV/
        ├── GAN/GAN.ipynb
        └── CGAN/CGAN.ipynb
```

### Synthetic tables

`SDV_Synthetic.ipynb` fits GaussianCopula, CTGAN, and TVAE on the real dataset
and saves each synthesizer as a `.pkl` beside the notebook. The fitted `.pkl`
files are not committed; download them from the Drive folder linked in
`Datasets/readme.md` to skip the fit. `gen_synth.py` then samples those
synthesizers at 10k / 100k / 1M rows:

```bash
.venv/Scripts/python -I DataGeneration/SyntheticData/gen_synth.py 10000 100000 1000000
```

It writes into each dataset's folder under `Datasets/`, skips files that already
exist, and is resumable. SDV must stay on 1.x — see `SETUP.md`.

GAN and CGAN are conditional generators: the class label is one-hot concatenated
to the noise vector, sized from the dataset's class count, so the same notebook
serves the binary UAV-ATTACK data and the four-class 2D data. GAN draws a fixed
10,000 samples with labels sampled uniformly; CGAN generates per class, matching
the training set's class counts.

Every generator writes through the canonical dataset schema, so a generated
table is indistinguishable from the real data in column names, column order, and
label encoding.

### Binary images

`tabToImage.ipynb` renders one image per row: each feature becomes a row of bit
pixels, padded to at least `precision` rows. UAV-ATTACK yields 82x64 images
(82 features, 64-bit encoding) and 2D yields 64x64. Images land in
`image-results-spoofing/data/Train/<class>/`.

`split_images.py` then moves the test split into `.../data/Test/<class>/` and
records membership in `split_manifest.json`. It is safe to re-run: it adopts a
pre-split archive (the UAV-ATTACK images ship split 2,897/725), completes a
legacy manifest, and refuses to split an already-split pool.

The downloadable image archives are linked from `Datasets/readme.md`. Extract
each into its dataset's `image-results-spoofing/` folder if you would rather not
render 510,530 images locally.

### Where the behaviour lives

Both halves are thin callers of `gps_ids/`:

- `gps_ids/schema.py` — the canonical columns and label vocabulary every
  generated table is normalized to.
- `gps_ids/images.py` — rendering, folder organization, the idempotent split,
  the image shape derived from the schema, and the loaders the image models use.

Change the bit precision, the split ratio, or the class folders there, not in
these notebooks.
