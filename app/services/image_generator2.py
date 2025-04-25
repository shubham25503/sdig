from PIL import Image
from io import BytesIO
from model.sd_model1 import pipeline
import torch

# Base prompts for Botox and Filler treatments (optimized)
base_prompts = {
    # Botox areas
    "forehead_lines_botox": "Smooth horizontal forehead lines while maintaining natural skin texture, tone, and expressions. Subtle, realistic improvement without altering facial identity.",
    "frown_lines_glabella_botox": "Reduce vertical '11' lines between eyebrows, keeping a relaxed, natural look. Preserve muscle balance and skin realism.",
    "crows_feet_botox": "Softly diminish crow’s feet around the eyes while preserving eye shape and natural expressions. Maintain fine skin texture.",
    "nasalis_lines_botox": "Soften nasal 'bunny' lines, retaining natural nose contours and realistic skin appearance.",
    "vertical_lip_lines_botox": "Subtly smooth vertical wrinkles above the lips while preserving natural lip texture, curves, and surrounding skin.",
    "lip_flip_botox": "Enhance the upper lip’s fullness with a natural lift near Cupid’s Bow. Preserve lip shape, texture, and volume.",
    "smile_lift_botox": "Lift corners of the mouth slightly, reducing downward smile lines naturally. Maintain smile structure and facial harmony.",
    "masseter_reduction_botox": "Slightly slim the jawline by softening the masseter muscles while preserving facial symmetry and jaw contours.",
    "dimpled_chin_botox": "Smooth dimpled chin texture while keeping natural chin definition and facial proportions.",
    "platysmal_bands_botox": "Reduce vertical neck bands, creating a smoother neckline while preserving skin texture and natural contours.",

    # Filler areas
    "cheek_filler": "Add gentle volume to the cheeks with natural, lifted facial contours. Preserve skin texture, symmetry, and balance.",
    "smile_line_filler": "Subtly fill nasolabial folds (smile lines) for a smoother, youthful look while keeping natural facial movement and expressions.",
    "lip_filler": "Plump and naturally shape the lips with soft, balanced volume enhancement. Maintain lip texture and proportions.",
    "temple_filler": "Restore lost volume in the temples for a refreshed, youthful contour while preserving natural facial lines and textures.",
    "nose_filler": "Smooth and refine the nasal bridge and tip with subtle, natural contour improvements. Maintain original nose shape."
}

# Max units for Botox areas
max_units = {
    "forehead_lines_botox": 30,
    "frown_lines_glabella_botox": 25,
    "crows_feet_botox": 30,
    "nasalis_lines_botox": 15,
    "vertical_lip_lines_botox": 8,
    "lip_flip_botox": 6,
    "smile_lift_botox": 12,
    "masseter_reduction_botox": 60,
    "dimpled_chin_botox": 8,
    "platysmal_bands_botox": 30
}

# Define areas
BOTOX_AREAS = set(max_units.keys())
FILLER_AREAS = set(area for area in base_prompts if area not in BOTOX_AREAS)


def get_ordinal_suffix(n: int) -> str:
    """Returns ordinal suffix (st, nd, rd, th) for a number."""
    return "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def build_prompt(area: str, injection_number: int) -> (str, float):
    """Builds the prompt and strength based on treatment type and units."""
    treatment_name = area.replace("_", " ").title()

    if area in BOTOX_AREAS:
        max_area_units = max_units.get(area, 30)
        normalized_units = min(injection_number / max_area_units, 1.0)
        effect_strength = 0.1 + (0.3 * (normalized_units ** 0.7))
        strength = min(effect_strength, 0.25)  # Slightly lower max cap for safety

        # prompt = (
        #     f"High-quality medical photograph after {injection_number} units of Botox in the {treatment_name} area. "
        #     f"{base_prompts.get(area, '')} No distortion, no facial identity changes."
        # )
        prompt = (
        f"High-quality medical photograph after {injection_number} units of Botox in the {treatment_name} area. "
        f"{base_prompts.get(area, '')} Eyes, facial features, and skin tone remain completely unchanged. "
        "No artistic changes. Strictly realistic and medically accurate."
        )   


    elif area in FILLER_AREAS:
        strength = 0.25  # Filler generally uses fixed, subtle strength
        prompt = (
            f"High-quality medical photograph after filler treatment in the {treatment_name} area. "
            # f"{base_prompts.get(area, '')} No distortion, no facial identity changes."
            f"{base_prompts.get(area, '')} Eyes, facial features, and skin tone remain completely unchanged. "
            "No artistic changes. Strictly realistic and medically accurate."
        )

    else:
        raise ValueError(f"Unknown treatment area: {area}")

    return prompt, strength, treatment_name


def generate_images(image: Image.Image, area: str, injection_number: int = 0):
    """Generates enhanced images based on treatment type and injection units."""
    results = []
    prompt, strength, treatment_name = build_prompt(area, injection_number)

    # Strong, clean negative prompt to eliminate distortions
    negative_prompt = (
    "extra fingers, mutated hands, blurry, deformed, bad anatomy, disfigured, poorly drawn face, mutation, "
    "fused fingers, too many fingers, long neck, cloned face, duplicate face, alien, plastic, waxy, cartoon, "
    "unnatural skin, unnatural eye color, changed eye color, face distortion, identity change, glowing skin, anime"
    )

    print(f"Generating {treatment_name} result"
          f"{' with ' + str(injection_number) + ' units' if area in BOTOX_AREAS else ''}...")

    try:
        # Generate the image using the pipeline
        output_image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=8.5  # Slightly higher for better prompt adherence
        ).images[0]

        # Save the result into a buffer
        buffer = BytesIO()
        output_image.save(buffer, format="JPEG")
        buffer.seek(0)

        results.append({
            "area": treatment_name,
            "image_bytes": buffer.getvalue()
        })
    except Exception as e:
        print(f"Error during image generation: {e}")
    finally:
        # Clear GPU memory after each execution
        torch.cuda.empty_cache()

    return results
