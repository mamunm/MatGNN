"""Download the bulk data from pymatgen database."""

from pathlib import Path

from ase import Atoms, db
from jarvis.db.figshare import data
from loguru import logger


def download_c2db_data() -> None:
    """Download the bulk data from jarvis database."""

    c2db_data = data("c2db")
    db_path = Path("c2db_data.db")
    if db_path.exists():
        db_path.unlink()

    logger.info("Downloading bulk data for C2DB materials...")

    with db.connect(db_path) as ase_db:
        for datum in c2db_data:
            mol = Atoms(
                cell=datum["atoms"]["lattice_mat"],
                positions=datum["atoms"]["coords"],
                pbc=[True, True, True],
                symbols=datum["atoms"]["elements"],
            )

            ase_db.write(
                mol,
                gap=datum["gap"],
                etot=datum["etot"],
                wf=datum["wf"],
                efermi=datum["efermi"],
            )

    logger.info("C2DB data downloaded.")


if __name__ == "__main__":
    download_c2db_data()
