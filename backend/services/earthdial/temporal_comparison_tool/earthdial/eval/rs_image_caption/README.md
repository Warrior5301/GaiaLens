# 📌 Evaluation Pipeline for Image Captioning

## 🌟 Overview
This repository provides a streamlined **evaluation pipeline** for the **Image Captioning** task.

---


## 📥 Download EarthDial Image Captioning Dataset

The **EarthDial-Dataset** is hosted on the [Hugging Face Hub](https://huggingface.co/datasets/akshaydudhane/EarthDial-Dataset). 
Dataset structure look like

```bash
EarthDial_downstream_task_datasets/
│
├── Classification/
│   ├── AID/
│   │   └── test/data-00000-of-00001.arrow
│   └── ...
│
├── Detection/
│   ├── NWPU_VHR_10_test/
│   ├── Swimming_pool_dataset_test/
│   └── ...
│
├── Region_captioning/
│   └── NWPU_VHR_10_test_region_captioning/
│
├── Image_captioning/
│   ├── RSICD_Captions/
│   └── UCM_Captions/
│...
```

You can download it using the `huggingface_hub` Python package.


### Requirements

Install the required package:

```shell
pip install huggingface_hub
```

### Download Instructions

Before downloading the datasets, ensure that the following directory structure exists:

```shell
mkdir -p src/earthdial/eval/Eardial_downstream_task_datasets
```
Use the following Python script to download only the `Image_captioning` subfolder:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="akshaydudhane/EarthDial-Dataset",
    repo_type="dataset",
    allow_patterns="Eardial_downstream_task_datasets/Image_captioning/**",
    local_dir=""
)
````

### Output
The dataset will be saved in a local directory named Eardial_downstream_task_datasets/Image_captioning, preserving the internal folder structure. After preparation, the expected directory structure will be:


After preparation, the expected directory structure will be:

```shell
EarthDial/Eardial_downstream_task_datasets/Image_captioning/
 ├── NWPU_RESISC45_Captions
 ├── RSICD_Captions
 ├── RSITMD_Captions
 ├── sydney_Captions
 ├── UCM_Captions
```
Notes
This dataset is intended for evaluation only and does not include predefined train/val/test splits. All files are in .arrow format and can be read using libraries like datasets or pyarrow.


## 📦 Download EarthDial Model Checkpoints

The EarthDial models are available on the Hugging Face Hub.

[EarthDial_4B_RGB](https://huggingface.co/akshaydudhane/EarthDial_4B_RGB)


### 🧩 Download Instructions

You can download them using the `huggingface_hub` Python package.
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="akshaydudhane/EarthDial_4B_RGB",
    repo_type="model",
    local_dir="checkpoints/EarthDial_4B_RGB"
)
```
---

## 🚀 Running the Evaluation

To execute the evaluation process on an **8-GPU setup**, run the following command:

```shell
# Test the rs_image_caption datasets
GPUS=8 ./src/earthdial/eval/eval.sh rs_image_caption --dynamic
```

This tests our EarthDial-4B on for image captioning task, saves result files (e.g., `src/earthdial/eval/rs_image_caption/results/NWPU_RESISC45_Captions.jsonl`) and displays the captioning score.

---

## 📌 Notes
- Ensure all dependencies are installed before running the scripts.
- Modify the script parameters if needed to match your system's configuration.
- Contributions & improvements are welcome! 🚀

