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

## Citation

TODO: add citation once paper is published.
