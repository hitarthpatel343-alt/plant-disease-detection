
import re
import requests
from io import BytesIO
from pypdf import PdfReader

SOURCE_URL = (
    "https://ppqs.gov.in/sites/default/files/"
    "pop-apple_final-_08.12.2022.pdf"
)


def get_online_disease_info(disease_name):
    if disease_name != "Apple___Apple_scab":
        return {
            "found": False,
            "message": (
                "Online information is not configured "
                "for this disease yet."
            ),
            "source": SOURCE_URL,
        }

    print("\nRetrieving official agricultural information...")

    try:
        response = requests.get(SOURCE_URL, timeout=30)
        response.raise_for_status()

        reader = PdfReader(BytesIO(response.content))

        # Find the page containing the Apple Scab heading.
        start_page = None
        page_texts = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            page_texts.append(text)

            normalized = " ".join(text.lower().split())

            if (
                "12.1 apple scab" in normalized
                and "venturia inaequalis" in normalized
            ):
                start_page = page_number
                break

        if start_page is None:
            return {
                "found": False,
                "message": "Apple Scab section was not found.",
                "source": SOURCE_URL,
            }

        # Read the following pages until the next disease heading.
        section = "\n".join(page_texts[start_page:])

        next_heading = re.search(
            r"12\.2\s+Premature\s+Leaf\s+Fall",
            section,
            flags=re.IGNORECASE,
        )

        if next_heading:
            section = section[:next_heading.start()]

        # Remove common PDF layout artifacts.
        section = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", section)
        section = re.sub(r"[ \t]+", " ", section)
        section = re.sub(r"\n{3,}", "\n\n", section)

        # Extract sections using their headings.
        def extract_section(label, next_labels):
            pattern = (
                rf"{label}\s*:\s*(.*?)"
                rf"(?=\n\s*(?:{'|'.join(next_labels)})\s*:|\Z)"
            )

            match = re.search(
                pattern,
                section,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if not match:
                return ""

            return match.group(1).strip()

        symptoms = extract_section(
            "Symptoms",
            ["Disease Development", "Management"],
        )

        development = extract_section(
            "Disease Development",
            ["Management"],
        )

        management = extract_section(
            "Management",
            [],
        )

        # Split management text into readable bullet points.
        management_steps = re.findall(
            r"[•]\s*(.*?)(?=(?:[•]\s*)|\Z)",
            management,
            flags=re.DOTALL,
        )

        management_steps = [
            re.sub(r"\s+", " ", item).strip()
            for item in management_steps
            if item.strip()
        ]

        return {
            "found": True,
            "disease": "Apple Scab",
            "scientific_name": "Venturia inaequalis",
            "symptoms": symptoms,
            "disease_development": development,
            "management": management_steps,
            "source": SOURCE_URL,
        }

    except requests.RequestException as error:
        return {
            "found": False,
            "message": f"Could not access the source: {error}",
            "source": SOURCE_URL,
        }

    except Exception as error:
        return {
            "found": False,
            "message": f"Could not process the document: {error}",
            "source": SOURCE_URL,
        }


if __name__ == "__main__":
    result = get_online_disease_info("Apple___Apple_scab")
    print(result)

