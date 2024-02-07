import json
from pathlib import Path
from typing import Any, Tuple

import requests
from loguru import logger

GRAPHQL = "http://api.catalysis-hub.org/graphql"


def fetch(query: str) -> Any:
    """Fetch data from the catalysis-hub GraphQL API."""
    return requests.get(GRAPHQL, {"query": query}).json()["data"]


def get_idx() -> int:
    """Get the current index of the catalysis-hub GraphQL API."""
    data_path = Path("raw_json_data")
    if data_path.exists():
        files = list(data_path.glob("*.json"))
        return len(files)
    else:
        return 0


def get_state() -> Tuple[bool, str]:
    """Get the current state of the catalysis-hub GraphQL API."""
    state_path = Path("state.json")
    if state_path.exists():
        with open(state_path, "r") as f:
            data = json.load(f)
            return data["has_next_page"], data["start_cursor"]
    else:
        return True, ""


def save_state(has_next_page: bool, start_cursor: str) -> None:
    """
    Save the current state of the catalysis-hub GraphQL API.

    Args:
        has_next_page (bool): Whether or not there are more reactions to fetch.
        start_cursor (str): The start cursor for the next page of reactions.
    """
    with open("state.json", "w") as f:
        json.dump({"has_next_page": has_next_page, "start_cursor": start_cursor}, f)


def reactions_from_pub(pub_id: str, page_size: int = 10) -> None:
    """
    Fetch reactions for the publication from the catalysis-hub GraphQL API.

    Args:
        pub_id (str): The publication ID.
        page_size (int, optional): The number of reactions to fetch. Defaults to 100.
    """
    data_path = Path("raw_json_data")
    if not data_path.exists():
        data_path.mkdir()
    has_next_page, start_cursor = get_state()
    while has_next_page:
        data = fetch(
            """{{
            reactions(pubId: "{pub_id}",
            first: {page_size},
            after: "{start_cursor}") {{
                totalCount pageInfo {{
                    hasNextPage
                    hasPreviousPage
                    startCursor
                    endCursor
                    }}
                edges {{
                    node {{
                        Equation
                        reactants
                        products
                        sites
                        reactionEnergy
                        reactionSystems {{
                            name
                            systems {{
                                energy
                                InputFile(format: "json")
                                }}
                    }}
                }}
            }}
        }}
    }}""".format(
                start_cursor=start_cursor,
                page_size=page_size,
                pub_id=pub_id,
            )
        )
        logger.info(f"Fetched reactions for start_cursor = {start_cursor}.")
        has_next_page = data["reactions"]["pageInfo"]["hasNextPage"]
        start_cursor = data["reactions"]["pageInfo"]["endCursor"]
        save_state(has_next_page, start_cursor)
        reactions = list(map(lambda x: x["node"], data["reactions"]["edges"]))
        idx = get_idx()
        with open(data_path / f"reactions_{idx + 1}.json", "w") as f:
            json.dump(reactions, f)


if __name__ == "__main__":
    reactions_from_pub("MamunHighT2019")
