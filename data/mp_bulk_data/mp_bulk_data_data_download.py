"""Download the bulk data from pymatgen database."""

import json
from pathlib import Path

import pandas as pd
from ase import db, io
from loguru import logger
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

# API key to access the MPRester
API_KEY = "mv5YUWkC6EkANiJVUNFWTCbKseyFXXg0"


def download_mp_bulk_data() -> None:
    """Download the bulk data from pymatgen database."""

    mat_ids = pd.read_csv("mp-ids-46744.csv", header=None)[0].to_list()
    n_mat_ids = len(mat_ids)
    mat_ids.sort(key=lambda x: int(x.split("-")[1]))
    mat_ids = list(
        (mat_ids[1000 * i : 1000 * (i + 1)] for i in range(0, 1 + len(mat_ids) // 1000))
    )
    Path("DATA").mkdir(exist_ok=True)
    db_path = Path("mp_bulk_data.db")
    if db_path.exists():
        db_path.unlink()

    logger.info(f"Downloading bulk data for {n_mat_ids} materials...")
    for i, mat_id in enumerate(mat_ids):
        with MPRester(API_KEY) as mpr:
            temp_data = mpr.summary.search(
                material_ids=mat_id, fields=["structure", "formation_energy_per_atom"]
            )

        formation_energy = []
        with db.connect(db_path) as ase_db:
            for m_id, out in zip(mat_id, temp_data):
                cif = CifWriter(out.structure)
                cif.write_file(f"DATA/{m_id}.cif")
                formation_energy.append((m_id, out.formation_energy_per_atom))
                mol = io.read(f"DATA/{m_id}.cif")
                ase_db.write(mol, hof=out.formation_energy_per_atom)

        with open(
            f"DATA/formation_energy_{i}_{mat_id[0]}_{mat_id[-1]}.json", "a+"
        ) as f:
            json.dump(formation_energy, f)
    logger.info("Bulk data downloaded.")
    logger.info("Please see DATA folder for the bulk data.")


if __name__ == "__main__":
    download_mp_bulk_data()
