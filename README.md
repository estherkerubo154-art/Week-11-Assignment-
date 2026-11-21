# Week 11 Assignment — Running a Machine Learning Job on HPC

**Student:** Esther Kerubo  
**Project:** Diabetes Risk Prediction  
**Dataset:** Pima Indians Diabetes Database  
**Models:** Logistic Regression and Random Forest Classifier

## 1. Files included

- `diabetes_risk_hpc.py` — loads the dataset, preprocesses features, trains and evaluates both classifiers, and saves the best model.
- `diabetes_risk_job.slurm` — Slurm batch script requesting HPC resources.
- `requirements.txt` — required Python packages.
- `README.md` — setup and assignment explanation.

## 2. Access the HPC environment

Use the login address and account supplied during the KENET HPC session. A typical SSH command is:

```bash
ssh YOUR_USERNAME@HPC_LOGIN_ADDRESS
```

The actual address, account, authentication, and scheduler settings are institution-specific. Do not put passwords, private keys, or access tokens in GitHub.

## 3. Set up the working directory and Python environment

```bash
mkdir -p ~/week11_diabetes_risk
cd ~/week11_diabetes_risk
```

Copy the assignment files into this directory. Create and activate a virtual environment:

```bash
python3 -m venv ~/venvs/diabetes-ml
source ~/venvs/diabetes-ml/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the HPC centre requires modules or a managed environment, follow the instructions from the KENET session instead.

## 4. Provide the dataset

Place the Pima diabetes CSV in the working directory. The program accepts a CSV with an `Outcome` column or the standard nine-column Pima CSV without a header. If no local CSV exists, it attempts to download the public dataset from the source URL in the Python script.

For reliable batch execution, copy the dataset to the working directory before submitting the job. Compute nodes may not have unrestricted internet access.

The expected predictors are Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age. The target is `Outcome` (0 = no diabetes, 1 = diabetes).

The script treats zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI as missing measurements and imputes them using training-set medians inside each model pipeline. Pregnancies is not treated this way.

## 5. Submit the Slurm job

Ensure that the virtual-environment activation line in `diabetes_risk_job.slurm` is uncommented and points to your environment. Apply any Python module or partition requirements stated by the HPC administrator.

Submit:

```bash
sbatch diabetes_risk_job.slurm
```

The scheduler will return a job ID, for example `Submitted batch job 123456`. This is an illustration only; use your actual job ID.

## 6. Check status and retrieve output

```bash
squeue -u "$USER"
```

When the job has finished, inspect the scheduler log files:

```bash
cat diabetes_ml_JOB_ID.out
cat diabetes_ml_JOB_ID.err
```

Replace `JOB_ID` with the actual number. A successful run should print the evaluation table and `JOB COMPLETED SUCCESSFULLY`.

The program saves:
- `outputs/model_performance.csv`
- `outputs/diabetes_risk_best_model.joblib`

Retrieve these files using the transfer method demonstrated during the HPC session, such as `scp` or the centre's approved transfer service.

## 7. Computing resources requested

| Resource | Request | Reason |
|---|---:|---|
| Nodes | 1 | The dataset and training workflow fit on one node. |
| Tasks | 1 | A single Python process performs the workflow. |
| CPUs per task | 2 | The Random Forest can use two worker threads; two CPUs are adequate for this small tabular dataset. |
| Memory | 4 GB | A reasonable starting allocation for the dataset, preprocessing, and model fitting. |
| Wall time | 30 minutes | A reasonable initial limit for data loading and training both models. |

These are starting allocations, not measured peak usage. Adjust them only after checking scheduler accounting or resource-limit messages. Follow any cluster-specific limits and directives.

## 8. How HPC differs from Colab or a personal computer

- **HPC:** The user submits a batch job to a scheduler. It may wait in a queue until the requested resources are available. The scheduler allocates CPUs, memory and time, and the job writes logs to files.
- **Google Colab:** The user commonly runs code interactively in a hosted notebook session. Runtime limits and availability are managed by Colab, and sessions may disconnect.
- **Personal computer:** Code runs directly on local hardware and uses that computer's available resources. It normally has no shared cluster scheduler or queue.

HPC is useful for larger datasets and computationally demanding experiments. This diabetes example is relatively small; the assignment demonstrates how to prepare, submit, monitor and retrieve a scheduled job.

## 9. Evidence of successful execution

The assignment requires evidence from the actual HPC system. After you submit the job, include:
1. A screenshot showing the `sbatch` command and returned job ID.
2. A screenshot or terminal output showing job status.
3. The `diabetes_ml_JOB_ID.out` log showing the metrics and `JOB COMPLETED SUCCESSFULLY`.
4. The generated `outputs/model_performance.csv`.

The job has not been run on your KENET account by preparing these files. Do not submit invented job IDs or fabricated screenshots.

## 10. Interpretation and limitations

The model is a learning exercise, not a medical diagnostic tool. The Pima dataset represents a particular study population and may not generalize to other populations or clinical settings. Evaluate precision, recall, F1 and ROC-AUC alongside accuracy, because class errors have different consequences. Any practical use would require independent clinical validation, privacy safeguards, bias assessment and professional oversight.

## 11. GitHub submission

Upload the Python script, Slurm job file, requirements file, README and (where dataset terms permit) the performance CSV and job log after a successful run. Do not upload passwords, private keys or restricted data.
