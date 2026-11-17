# Situational Awareness Dataset & SVM Baseline

This repository contains the raw eye-tracking data, processed gaze features, and SVM training code accompanying our paper.

---

## Directory Structure

```
code_release/
├── raw_data/
│   ├── transcriptions.csv      # Main annotation file (3320 trials, 43 subjects)
│   ├── queries.csv             # Stimulus query metadata (143 driving scenarios)
│   └── gaze_data/              # Per-trial raw eye-tracking CSVs (3320 files)
│       └── subj_XX-stimulus_YYY-Z.csv
├── processed_data/             # Per-trial extracted gaze features (3320 files)
│   └── {trial_index}.csv
├── read_data.py                # Processes raw_data → processed_data
├── train_svm.py                # Trains and evaluates SVM classifiers
└── README.md
├── LICENSE-CODE                # CC BY-NC 4.0 (code)
├── LICENSE-DATA                # CC BY-NC-SA 4.0 (data)
```

---

## Data Description

### `raw_data/transcriptions.csv`
The main experimental annotation file. Each row is one subject's response to a stimuli. The data is already transcribed into text, processed through the VLM, and manually corrected to get the final L1, L2 and L3 awareness labels.

| Column | Description |
|---|---|
| `seq_id` | Unique trial index |
| `subject_id` | Participant ID (`subj_00` – `subj_42`) |
| `stimulus_id` | Stimulus identifier |
| `stimulus_timing` | Timing condition of the stimulus |
| `video_file` | Waymo video clip filename (used to look up Waymo parquet data) |
| `audio_file` | Path to participant's spoken response audio |
| `question_image` | Filename of the question frame shown to the participant |
| `transcription` | Transcribed verbal response |
| `gaze_file_path` | Relative path to the gaze CSV in `raw_data/gaze_data/` |
| `frame_start` | Video frame at which the trial began |
| `frame_end` | Video frame at which the question was asked |
| `L1_existence_flag` | Final VLM correct and manual screened label: participant identified the object existed (0/1) |
| `L2_context_flag` | Final VLM correct and manual screened label: participant described the object's current context with respect the other objects and the driving scenario (0/1) |
| `L3_future_flag` | Final VLM correct and manual screened label: participant predicted the object's future action (0/1) |

**43 participants, 143 unique stimuli, 3320 trials total.**
Label distribution: L1 81.5% positive, L2 79.3% positive, L3 56.2% positive.
Labels are hierarchical: L3=1 implies L2=1 implies L1=1.

### `raw_data/queries.csv`
Metadata about each of the 143 driving scenarios used as stimuli. One row per scenario.

| Column | Description |
|---|---|
| `Videos` | Waymo video clip filename |
| `object_id` | Waymo object UUID of the target object |
| `Start_Time` | Time (s) the target object becomes relevant |
| `End_Time` | Time (s) at which the question is asked |
| `Answer` | Full ground-truth SA description |
| `Existence` | Ground-truth L1 answer phrase |
| `Context` | Ground-truth L2 answer phrase |
| `Future` | Ground-truth L3 answer phrase |
| `Query Type` | Motion category (`Lane Change`, `Turn`, `Cross`, `Pass`, `Brake`, `Enter/exit vehicle`) |

### `raw_data/gaze_data/`
One CSV per trial, named `subj_XX-stimulus_YYY-Z.csv`. Each row is one gaze sample.

| Column | Description |
|---|---|
| `video_ts` | Timestamp in milliseconds relative to trial start |
| `lx`, `ly` | Left-eye gaze position (normalized 0–1, screen coordinates) |
| `rx`, `ry` | Right-eye gaze position (normalized 0–1, screen coordinates) |
| `frame_id` | Waymo video frame filename aligned to this sample |

Rows with both eyes missing are sparse (tracker loss). `read_data.py` averages left/right when both are valid, falls back to one eye when the other is NaN.

### `processed_data/`
One CSV per trial (indexed by row position in `transcriptions.csv`). Each file has exactly one row of extracted features.

| Column | Description |
|---|---|
| `q_image` | Question image filename (links back to `transcriptions.csv`) |
| `gaze_dist_pause` | Gaze distance to object at the final frame (pixels) |
| `gaze_dist_avg` | Mean gaze distance to object across all fixation frames |
| `gaze_dist_min` | Minimum gaze distance to object (closest approach) |
| `contrasts` | Delta-E (CIE76) color contrast of object vs. background |
| `object_size` | Bounding box area of object at question frame (pixels²) |
| `object_distance` | Object distance from image center (pixels) |
| `object_duration` | Number of frames in which the object was visible |
| `h_avg_{1–4}` | Mean gaze distance within each of 4 visual-angle radii (2.5°, 4.1°, 9.1°, 15.0°) |
| `h_dwell_time_{1–4}` | Frame count where gaze fell within each radius |
| `h_elapse_time_{1–4}` | Time elapsed (µs) from last in-radius gaze to trial end |

---

## Setup

```bash
pip install pandas numpy scikit-learn scikit-image opencv-python pillow
```

> **Note:** `read_data.py` additionally requires access to the [Waymo Open Dataset v2.0.1](https://waymo.com/open/) parquet files (`camera_image` and `camera_box` splits) for the training sequences. The processed features in `processed_data/` were derived from that data and are included here so the SVM training (`train_svm.py`) can be run without the Waymo data.

---

## Usage

### Step 1: Process raw gaze data (requires Waymo parquets)

```bash
python read_data.py
```

Reads `raw_data/transcriptions.csv` and `raw_data/gaze_data/`, aligns gaze to Waymo video frames, extracts features, and writes one CSV per trial to `processed_data/`.

### Step 2: Train SVM classifiers

```bash
python train_svm.py
```

Reads all CSVs from `processed_data/` and `raw_data/transcriptions.csv`. Runs 3 trials of stratified 80/20 train/test splits and evaluates four classifiers: (1) three independent binary RBF-SVMs (one per SA level), (2) a cascaded binary classifier that combines existence and future probabilities into a joint 3-class distribution, (3) a hierarchical binary classifier where each level's predicted probability is appended as a feature to the next level's classifier, and (4) a simple 3-class multiclass SVM. Reports mean ± std of accuracy, UAR, precision, recall, F1, and ROC AUC across all methods.

---

## License

This repository uses two licenses:

| Material | License | File |
|---|---|---|
| Code (`read_data.py`, `train_svm.py`) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | [`LICENSE-CODE`](LICENSE-CODE) |
| Data (`raw_data/`, `processed_data/`) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | [`LICENSE-DATA`](LICENSE-DATA) |

Both licenses prohibit commercial use and require attribution. The data license additionally requires that derivative datasets be shared under the same license. If you use this repository, please cite our paper (see [Citation](#citation)).

### Waymo Open Dataset

This repository does **not** contain or redistribute Waymo Open Dataset data (no camera images, boxes, or labels). It contains annotations collected by us (participant transcriptions, awareness labels, and eye-tracking recordings) together with features and metadata that reference or were computed from the Waymo Open Dataset:

- `raw_data/queries.csv` and `raw_data/transcriptions.csv` refer to Waymo video clips and object IDs.
- `processed_data/` contains per-trial features (for example, object size, gaze distance to the object, and color contrast) computed from Waymo Open Dataset v2.0.1 camera images and bounding boxes.

The CC licenses above do not replace or override the [Waymo Open Dataset License Agreement](https://waymo.com/open/terms), which continues to apply to the Waymo-derived materials listed here. In particular:

- You must obtain the Waymo Open Dataset yourself, from [waymo.com/open](https://waymo.com/open), and accept Waymo's terms to run `read_data.py` or to link our annotations to the original video.
- Waymo-derived materials may be used for non-commercial purposes only (research, teaching, and scientific publication), and may not be used to operate or assist in operating a vehicle or in production systems.
- Work that uses Waymo-derived materials should credit the Waymo Open Dataset and reference its terms, as the Waymo agreement requires.

**Attribution.** This dataset and its accompanying code were made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at [waymo.com/open/terms](https://waymo.com/open/terms), and your access and use of such work are governed by the terms and conditions therein.

Waymo's suggested citation:

```bibtex
@misc{waymo_open_dataset,
    title   = {Waymo Open Dataset: An autonomous driving dataset},
    website = {\url{https://www.waymo.com/open}},
    year    = {2019-2025}
}
```

---

## Citation

If you use this dataset or code, please cite our paper:

```bibtex
@inproceedings{gupta2026see,
  title={See Something, Say Something: Layered Driver Situational Awareness from Language},
  author={Gupta, Pranay and Stent, Simon and Gideon, John and Battel, Kayli and Dees, Laporsha and Gomez, Patricio Reyes and Applegate-Kenton, Megan and Sumner, Emily and Rosman, Guy},
  booktitle={2026 IEEE Intelligent Vehicles Symposium (IV)},
  pages={1676--1683},
  year={2026},
  organization={IEEE}
}
```

Please also cite the Waymo Open Dataset (see [License](#waymo-open-dataset)).
