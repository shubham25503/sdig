
from PIL import Image
from io import BytesIO
from app.model.sd_model import pipeline


# Base prompts per treatment
base_prompts = {
    "forehead_botox": "Create realistic post-treatment image with smoother forehead lines proportional to injection units. Natural appearance, preserved skin texture.",
    "eye_botox": "Show realistic crow's feet reduction around eyes proportional to injection units. Natural eye shape, preserved expressions.",
    "glabella_botox": "Create after-image with reduced '11 lines' between eyebrows proportional to injection units. Natural muscle relaxation.",
    "jaw_botox": "Show natural jaw slimming from masseter relaxation proportional to injection units. Preserved facial proportions.",
    "bunny_lines_botox": "Render softened nasal lines proportional to injection units. Preserved natural expressions and nasal shape.",
    "chin_botox": "Show reduced chin dimpling proportional to injection units. Natural chin texture and shape."
}

def get_ordinal_suffix(n: int) -> str:
    return "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

def generate_images(image: Image.Image, area: str, injection_number: int):
    results = []
    # for area in areas:
    treatment_name = area.replace("_", " ").title()

    prompt = (
            f"Medical photo after {injection_number} units of Botox in {treatment_name} area. "
            f"{base_prompts.get(area, '')}"
    )

    negative_prompt = "unrealistic, cartoon, distorted features, changed identity"

    print(f"Generating {treatment_name} result with {injection_number} units...")

    # Dynamic strength based on injection efficacy curve
    # Calculate strength as a function of injection number relative to typical max dosage for the area
    max_units = {
            "forehead_botox": 30,
            "eye_botox": 30,
            "glabella_botox": 25,
            "jaw_botox": 60,
            "bunny_lines_botox": 15,
            "chin_botox": 8
    }

    # Calculate proportional strength based on area-specific dosage curves
    max_area_units = max_units.get(area, 30)
    normalized_units = min(injection_number / max_area_units, 1.0)
    # Apply a non-linear response curve (more realistic than linear)
    # This creates a more sigmoid-like response curve that matches clinical results
    effect_strength = 0.1 + (0.3 * (normalized_units ** 0.7))
    strength = min(effect_strength, 0.35)

    output_image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=7.5
    ).images[0]

    buffer = BytesIO()
    output_image.save(buffer, format="JPEG")
    buffer.seek(0)

    results.append({
        "area": treatment_name,
        "image_bytes": buffer.getvalue()
    })

    return results
