"""Create ASE DB from raw json data."""
import io
import json
from pathlib import Path
from typing import Any, List

from ase import db
from ase import io as ase_io
from loguru import logger

REACTIONS = [
    "0.5H2(g) + * -> H*",
    "H2O(g) - H2(g) + * -> O*",
    "H2O(g) - 0.5H2(g) + * -> OH*",
    "H2O(g) + * -> H2O*",
    "CH4(g) - 2.0H2(g) + * -> C*",
    "CH4(g) - 1.5H2(g) + * -> CH*",
    "CH4(g) - H2(g) + * -> CH2*",
    "CH4(g) - 0.5H2(g) + * -> CH3*",
    "0.5N2(g) + * -> N*",
    "0.5H2(g) + 0.5N2(g) + * -> NH*",
    "H2S(g) - H2(g) + * -> S*",
    "H2S(g) - 0.5H2(g) + * -> SH*",
]

ADSORBED_SPECIES = [
    "Hstar",
    "Ostar",
    "OHstar",
    "H2Ostar",
    "Cstar",
    "CHstar",
    "CH2star",
    "CH3star",
    "Nstar",
    "NHstar",
    "Sstar",
    "SHstar",
]

logger.add(
    "reaction_energy.log",
    colorize=False,
    backtrace=True,
    diagnose=True,
)


def get_reaction_data() -> List[Any]:
    reactions = []
    paths = Path("raw_json_data").glob("reactions_*")
    for path in paths:
        with open(path, "r") as f:
            reactions.extend(json.load(f))
    return reactions


reaction_data = get_reaction_data()

if Path("CatHub.db").exists():
    Path("CatHub.db").unlink()

with db.connect("CatHub.db") as ase_db:
    for i, reaction in enumerate(REACTIONS):
        for datum in reaction_data:
            try:
                if datum["Equation"].find(reaction) != -1:
                    if datum["sites"].find("top|") != -1:
                        system = [
                            d
                            for d in datum["reactionSystems"]
                            if d["name"] == ADSORBED_SPECIES[i]
                        ][0].pop("systems")
                        with io.StringIO() as tmp_file:
                            tmp_file.write(system.pop("InputFile"))
                            tmp_file.seek(0)
                            atoms = ase_io.read(tmp_file, format="json")
                        logger.info(f"reaction energy: {datum['reactionEnergy']}")
                        ase_db.write(atoms, reaction_energy=datum["reactionEnergy"])
                    elif datum["sites"].find("bridge|") != -1:
                        system = [
                            d
                            for d in datum["reactionSystems"]
                            if d["name"] == ADSORBED_SPECIES[i]
                        ][0].pop("systems")
                        with io.StringIO() as tmp_file:
                            tmp_file.write(system.pop("InputFile"))
                            tmp_file.seek(0)
                            atoms = ase_io.read(tmp_file, format="json")
                        logger.info(f"reaction energy: {datum['reactionEnergy']}")
                        ase_db.write(atoms, reaction_energy=datum["reactionEnergy"])
                    elif datum["sites"].find("hollow|") != -1:
                        system = [
                            d
                            for d in datum["reactionSystems"]
                            if d["name"] == ADSORBED_SPECIES[i]
                        ][0].pop("systems")
                        with io.StringIO() as tmp_file:
                            tmp_file.write(system.pop("InputFile"))
                            tmp_file.seek(0)
                            atoms = ase_io.read(tmp_file, format="json")
                        logger.info(f"reaction energy: {datum['reactionEnergy']}")
                        ase_db.write(atoms, reaction_energy=datum["reactionEnergy"])
            except Exception as e:
                print(e)
