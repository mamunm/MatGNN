"""Download the abse3 data from DTU physik website."""

from urllib.request import urlretrieve

from loguru import logger

# Link address to the database
LINK_ADDRESS = (
    "https://cmr.fysik.dtu.dk/_downloads/7df8781a8a82790b72fd2af57f8ce553/abse3.db"
)


def download_abse3_data() -> None:
    """Download the bulk data from DTU physik."""

    logger.info("Downloading bulk data for ABSe3 materials...")
    urlretrieve(LINK_ADDRESS, "abse3.db")
    logger.info("ABSe3 data downloaded.")


if __name__ == "__main__":
    download_abse3_data()
