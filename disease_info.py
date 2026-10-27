import json
from pathlib import Path

# Get the directory where this Python file is located
BASE_DIR = Path(__file__).resolve().parent

# Path to the disease database
DATABASE_PATH = BASE_DIR / "disease_database.json"


def load_disease_database():
    """Load disease information from the JSON database."""

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Disease database not found: {DATABASE_PATH}"
        )

    with open(DATABASE_PATH, "r", encoding="utf-8") as file:
        database = json.load(file)

    return database


def get_disease_info(disease_name):
    """Get information about a predicted plant disease."""

    database = load_disease_database()

    if disease_name not in database:
        return {
            "found": False,
            "message": "Information for this disease is not available."
        }

    return {
        "found": True,
        "information": database[disease_name]
    }


if __name__ == "__main__":

    # Load the database
    database = load_disease_database()

    print("Disease database loaded successfully!")
    print(f"Total disease records: {len(database)}")

    # Test with one disease
    disease_name = "Apple___Apple_scab"

    result = get_disease_info(disease_name)

    print("\nDisease Information:")
    print(json.dumps(result, indent=4))