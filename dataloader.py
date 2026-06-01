import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_DATASET = "montassarba/mimic-iv-clinical-database-demo-2-2"
DATA_DIR = "data"
EXTRACTED_DIR_NAME = "mimic-iv-clinical-database-demo-2.2"
BASE_PATH = os.path.join(DATA_DIR, EXTRACTED_DIR_NAME)

ALL_DATASETS = {
    "admissions": "hosp/admissions.csv",
    "patients": "hosp/patients.csv",
    "diagnoses_icd": "hosp/diagnoses_icd.csv",
    "d_icd_diagnoses": "hosp/d_icd_diagnoses.csv",
    "d_icd_procedures": "hosp/d_icd_procedures.csv",
    "drgcodes": "hosp/drgcodes.csv",
    "labevents": "hosp/labevents.csv",
    "pharmacy": "hosp/pharmacy.csv",
    "prescriptions": "hosp/prescriptions.csv",
    "procedures_icd": "hosp/procedures_icd.csv",
    "transfers": "hosp/transfers.csv",
    "icustays": "icu/icustays.csv",
    "chartevents": "icu/chartevents.csv",
    "datetimeevents": "icu/datetimeevents.csv",
    "inputevents": "icu/inputevents.csv",
    "outputevents": "icu/outputevents.csv",
    "procedureevents": "icu/procedureevents.csv",
}


def download_and_extract_mimic_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(BASE_PATH):
        print(f"[ok] Dataset already extracted at {BASE_PATH}")
        return

    zip_path = os.path.join(DATA_DIR, f"{KAGGLE_DATASET.split('/')[-1]}.zip")
    print("[..] Downloading MIMIC-IV dataset from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=False)

    print("[..] Extracting ZIP file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"[ok] Dataset extracted to {BASE_PATH}")


def load_all_mimic_datasets():
    download_and_extract_mimic_data()

    dataframes = {}
    print(f"[..] Loading CSVs from: {BASE_PATH}")

    for name, rel_path in ALL_DATASETS.items():
        full_path = os.path.join(BASE_PATH, rel_path)
        try:
            df = pd.read_csv(full_path)
            dataframes[name] = df
            print(f"[ok] Loaded '{name}' -> shape: {df.shape}")
        except FileNotFoundError:
            print(f"[!!] Missing: {rel_path}")
        except Exception as e:
            print(f"[!!] Error loading {name}: {e}")

    print(f"\nLoaded {len(dataframes)} datasets.")
    return dataframes


def load_icd_description_map(icd_df):
    return dict(zip(icd_df['icd_code'], icd_df['long_title']))


def prepare_patient_documents(data):
    patients = data.get("patients", pd.DataFrame())
    diagnoses_icd = data.get("diagnoses_icd", pd.DataFrame())
    d_icd_diagnoses = data.get("d_icd_diagnoses", pd.DataFrame())
    procedures_icd = data.get("procedures_icd", pd.DataFrame())
    d_icd_procedures = data.get("d_icd_procedures", pd.DataFrame())

    all_subjects = patients["subject_id"].unique() if "subject_id" in patients else []
    patient_docs = []

    for sid in all_subjects:
        fragments = []

        # Diagnoses
        dx = diagnoses_icd[diagnoses_icd["subject_id"] == sid]
        dx = dx.merge(d_icd_diagnoses, on="icd_code", how="left")
        dx_texts = dx["long_title"].dropna().unique().tolist() if "long_title" in dx.columns else []
        if dx_texts:
            fragments.append("Diagnoses: " + "; ".join(dx_texts))

        # ICU stays
        icu = data.get("icustays", pd.DataFrame())
        icu = icu[icu["subject_id"] == sid] if not icu.empty else pd.DataFrame()
        for _, row in icu.iterrows():
            fragments.append(f"ICU stay from {row['intime']} to {row['outtime']}, firstcare = {row['first_careunit']}")

        # Medications
        meds = data.get("prescriptions", pd.DataFrame())
        meds = meds[meds["subject_id"] == sid] if not meds.empty else pd.DataFrame()
        drug_names = meds["drug"].dropna().unique().tolist()
        if drug_names:
            fragments.append("Prescriptions: " + ", ".join(drug_names))

        # Procedures
        procs = procedures_icd[procedures_icd["subject_id"] == sid]
        procs = procs.merge(d_icd_procedures, on="icd_code", how="left")
        procs_texts = procs["long_title"].dropna().unique().tolist() if "long_title" in procs.columns else []
        if procs_texts:
            fragments.append("Procedures: " + "; ".join(procs_texts))

        # Lab Events (top 3 most recent)
        labs = data.get("labevents", pd.DataFrame())
        labs = labs[labs["subject_id"] == sid] if not labs.empty else pd.DataFrame()
        if not labs.empty:
            labs = labs.sort_values("charttime", ascending=False).head(3)
            lab_fragments = [f"{row['itemid']} = {row['value']}" for _, row in labs.iterrows()]
            if lab_fragments:
                fragments.append("Recent Labs: " + "; ".join(lab_fragments))

        if fragments:
            patient_summary = f"Subject {sid}\nSummary:\n" + "\n".join(fragments)
            patient_docs.append({"subject_id": str(sid), "text": patient_summary})

    print(f"Prepared {len(patient_docs)} patient documents.")
    return patient_docs
