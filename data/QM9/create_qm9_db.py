"""Create an ASE database from the raw xyz files."""


from io import StringIO
from pathlib import Path

from ase import db, io
from loguru import logger


def create_qm9_database() -> None:
    """Create a QM9 ASE database from the xyz data."""

    with db.connect("QM9_data.db") as ase_db:
        for xyz_file in Path("raw_data").glob("*.xyz"):
            content = xyz_file.read_text().split("\n")
            properties = content[1].split("\t")
            n_line = int(content[0]) + 2
            f = StringIO("\n".join(content[:n_line]))
            try:
                logger.info(f"Processing {xyz_file}.")
                mol = io.read(f, format="xyz")
            except ValueError:
                logger.info(f"{xyz_file} is not a valid xyz file.")
                continue
            ase_db.write(
                mol,
                RC_A=float(properties[1]),
                RC_B=float(properties[2]),
                RC_C=float(properties[3]),
                mu=float(properties[4]),
                alpha=float(properties[5]),
                homo=float(properties[6]),
                lumo=float(properties[7]),
                gap=float(properties[8]),
                r2=float(properties[9]),
                zpve=float(properties[10]),
                U_0=float(properties[11]),
                U_298=float(properties[12]),
                H_enthalpy=float(properties[13]),
                G_free_energy=float(properties[14]),
                Cv=float(properties[15]),
            )


if __name__ == "__main__":
    create_qm9_database()
