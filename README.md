# Pollen Prediction

This project implements a complete pipeline for airborne pollen forecasting using several time-series models. The pipeline evaluates both fully pre-trained foundational models (used directly for inference) and non-foundational models that require additional training/fine-tuning on the provided dataset.

## 📂 Directory Structure
- **datasets/** → Directory where the user can store the dataset (optional). The original dataset was removed to comply with confidentiality agreements.
- **notebooks/** → Contains all Jupyter notebooks used throughout the pipeline.
- **requirements/** → Includes several requirements files with dependencies needed for each model. They can be installed easily using `pip`.
- **src/** → Contains the `ppf` package, which groups Python modules with helper functions for the analysis.
- **outputs/** → Stores predictions, evaluations, execution times, and plots generated during the workflow.

## ⚙️ Setup

1. The main code is located in the `ppf` directory.  
2. Create a **virtual environment** for each model.  
3. Install a compatible **Python** version (tested with `Python 3.10.11`).  
4. Install a compatible **CUDA** version (tested with `CUDA 12.4`).  
5. Install a compatible **PyTorch** version. Example for Windows:  
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
6. Install the dependencies for each model (Moirai, Chronos, NHits, and NBeats) from the corresponding files in `requirements/`:
```bash
pip install -r requirements/<model_name>.txt
```
## Usage
1. Activate the virtual environment corresponding to the model you want to test.
2. Launch Jupyter Notebook or JupyterLab:
```bash
jupyter lab
```
3. Navigate to the `notebooks/` directory and open the notebook you want to run.
## Dataset
Some notebooks cannot be executed as-is because they relied on the original dataset (`AlnusOurense9322.csv`) or on predictions produced by models that were removed. The base code remains fully functional: you only need to add a new dataset with the same structure to run all notebooks again.

To use a different dataset:
1. Place the new dataset inside the `datasets/` directory.
2. If the dataset is already in CSV format, skip this step. If not, you have two options:
	1. Convert your dataset to CSV.
	2. Modify the code so that your dataset is loaded and stored as a DataFrame (instructions below).
3. In subsection 3.1.3 Get the pollen time series of the notebooks:
	- 02-01_Forecasting_with_Moirai
	- 02-02_Forecasting_with_Chronos
	- 02-03_Forecasting_with_NHiTs_one_year_train
	- 02-04_Forecasting_with_NHiTs_five_years_train
	- 02-05_Forecasting_with_NBeats_one_year_train
	- 02-06_Forecasting_with_NBeats_five_years_train
	Replace the dataset path with the path to your file. If stored in `datasets/` it should look like:

```python
df = pd.read_csv("../datasets/<file_name>.csv")
```
Alternatively, you may replace this line with code that converts your dataset into a DataFrame.

4. The expected dataset structure includes the following columns with daily samples from 1993 to 2023: `"date"`, `"pollen"`, `"rain"`, `"tmax"`, `"tmin"`, `"tmed"`.
	You may adapt these variables in subsections 3.1.1 Constants and 3.1.2 Arguments of the notebooks listed above.
	You are free to modify any function parameters across the notebooks. If you need to understand how the code works, refer to the source files in `src/ppf`.