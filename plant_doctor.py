
from predict import load_model, predict_disease
from online_disease_info import get_online_disease_info


def print_bullets(items):
    if not items:
        print("- Information was not available in the source.")
        return

    for item in items:
        print(f"- {item}")


def display_plant_information(disease_name, confidence):
    result = get_online_disease_info(disease_name)

    print("\n" + "=" * 55)
    print("              AI PLANT DOCTOR")
    print("=" * 55)

    print("\nPLANT DISEASE REPORT")
    print("-" * 55)
    print(f"Predicted class : {disease_name}")
    print(f"Model confidence: {confidence:.2f}%")

    if not result["found"]:
        print("\nDisease information is currently unavailable.")
        print(result["message"])
        print(f"\nSource: {result.get('source', 'Not available')}")
        return

    print(f"\nDisease         : {result['disease']}")
    print(f"Scientific name : {result['scientific_name']}")

    print("\n" + "-" * 55)
    print("1. WHAT IS THIS DISEASE?")
    print("-" * 55)

    print(
        f"{result['disease']} is a disease associated with "
        f"the fungus {result['scientific_name']}."
    )

    print("\n" + "-" * 55)
    print("2. SYMPTOMS")
    print("-" * 55)

    print(result["symptoms"] or "Symptoms were not extracted.")

    print("\n" + "-" * 55)
    print("3. HOW DOES THE DISEASE DEVELOP?")
    print("-" * 55)

    print(
        result["disease_development"]
        or "Disease development details were not extracted."
    )

    print("\n" + "-" * 55)
    print("4. MANAGEMENT STEPS FROM THE SOURCE")
    print("-" * 55)

    print_bullets(result["management"])

    print("\n" + "-" * 55)
    print("5. PRACTICAL NEXT STEPS")
    print("-" * 55)

    practical_steps = [
        "Inspect affected leaves and fruits carefully.",
        "Collect fallen leaves and manage plant debris "
        "according to local agricultural guidance.",
        "Prune appropriately to improve airflow and "
        "help foliage dry more quickly.",
        "Monitor the plant for new or worsening symptoms.",
        "Consult a local agricultural expert before "
        "using a chemical treatment.",
    ]

    print_bullets(practical_steps)

    print("\n" + "-" * 55)
    print("6. IMPORTANT NOTE")
    print("-" * 55)

    print(
        "The AI prediction may be incorrect. "
        "Model confidence is not a guarantee of correctness."
    )

    print("\nOfficial source:")
    print(result["source"])

    print("\n" + "=" * 55)


def main():
    try:
        model, class_names = load_model()

        image_path = input(
            "\nEnter the full path of a plant-leaf image: "
        ).strip().strip('"')

        results = predict_disease(
            image_path,
            model,
            class_names,
        )

        if not results:
            print("No predictions were returned.")
            return

        best_prediction = results[0]

        display_plant_information(
            best_prediction["disease"],
            best_prediction["confidence"],
        )

    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()

