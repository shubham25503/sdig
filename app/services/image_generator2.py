from PIL import Image
from io import BytesIO
from model.sd_model1 import pipeline
import torch




# Base prompts for Botox and Filler treatments (optimized)
base_prompts = {
    # Botox areas
    "forehead_lines_botox": "Smooth horizontal forehead lines while maintaining natural skin texture, tone, and expressions. Subtle, realistic improvement without altering facial identity.",
    "frown_lines_glabella_botox": "Reduce vertical '11' lines between eyebrows, keeping a relaxed, natural look. Preserve muscle balance and skin realism.",
    "crows_feet_botox": "Softly diminish crow's feet around the eyes while preserving eye shape and natural expressions. Maintain fine skin texture.",
    "nasalis_lines_botox": "Soften nasal 'bunny' lines, retaining natural nose contours and realistic skin appearance.",
    "vertical_lip_lines_botox": "Subtly smooth vertical wrinkles above the lips while preserving natural lip texture, curves, and surrounding skin.",
    "lip_flip_botox": "Enhance the upper lip's fullness with a natural lift near Cupid's Bow. Preserve lip shape, texture, and volume.",
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

# Common negative prompt elements that apply to all treatments
common_negative_prompt = (
    "changed face, changed skin tone, mutated hands, blurry, deformed, bad anatomy, disfigured, mutation, "
    "fused fingers, too many fingers, long neck, cloned face, duplicate face, alien, plastic, waxy, cartoon, "
    "unnatural skin, glowing skin, anime, identity change, poorly drawn face"
)

# Protection prompts for each facial area - used to prevent changes in non-target areas
area_protection_prompts = {
    "forehead": "changed forehead shape, unnatural forehead smoothness, frozen forehead, altered forehead proportions, changed forehead lines",
    "eyebrows": "changed eyebrow shape, uneven eyebrows, altered eyebrow position, raised eyebrows, lowered eyebrows",
    "eyes": "changed eye shape, different eye color, heterochromia, enlarged eyes, small eyes, squinted eyes, crossed eyes, asymmetric eyes, altered eye position",
    "nose": "changed nose shape, altered nostril size, uneven nostrils, changed nose position, altered nasal bridge, modified nose",
    "cheeks": "changed cheek volume, asymmetric cheeks, altered cheekbone height, changed cheek shape, overfilled cheeks",
    "lips": "changed lip shape, uneven lips, altered lip size, overfilled lips, duck lips, changed lip position, modified lip texture",
    "smile_lines": "changed nasolabial folds, asymmetric smile lines, altered smile line depth",
    "jaw": "changed jaw shape, asymmetric jaw, altered jaw angle, changed jawline, modified jaw width",
    "chin": "changed chin shape, altered chin projection, uneven chin, modified chin texture",
    "neck": "changed neck shape, altered neck muscles, modified neck texture, changed neck bands",
    "temples": "changed temple volume, asymmetric temples, altered temporal area",
    "facial_structure": "changed face shape, altered facial proportions, changed facial structure, asymmetric face, modified facial angles"
}




def get_protective_negative_prompt(target_area: str) -> str:
    """Generates a negative prompt that protects all areas except the target area."""
    # Map treatment areas to their core facial areas
    area_mapping = {
        "forehead_lines_botox": ["forehead"],
        "frown_lines_glabella_botox": ["forehead", "eyebrows"],
        "crows_feet_botox": ["eyes"],
        "nasalis_lines_botox": ["nose"],
        "vertical_lip_lines_botox": ["lips"],
        "lip_flip_botox": ["lips"],
        "smile_lift_botox": ["lips", "smile_lines"],
        "masseter_reduction_botox": ["jaw"],
        "dimpled_chin_botox": ["chin"],
        "platysmal_bands_botox": ["neck"],
        "cheek_filler": ["cheeks"],
        "smile_line_filler": ["smile_lines"],
        "lip_filler": ["lips"],
        "temple_filler": ["temples"],
        "nose_filler": ["nose"]
    }

    # Get the areas that should be allowed to change
    allowed_areas = area_mapping.get(target_area, [])
    
    # Combine protection prompts for all areas except the allowed ones
    protection_prompts = [
        prompt for area, prompt in area_protection_prompts.items()
        if area not in allowed_areas
    ]
    
    # Always protect overall facial structure unless specifically targeting it
    if "facial_structure" not in allowed_areas:
        protection_prompts.append(area_protection_prompts["facial_structure"])

    return ", ".join(protection_prompts)

def get_ordinal_suffix(n: int) -> str:
    """Returns ordinal suffix (st, nd, rd, th) for a number."""
    return "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def build_prompt(area: str, injection_number: int) -> (str, float):
    """Builds the prompt and strength based on treatment type and units."""
    treatment_name = area.replace("_", " ").title()

    if area in BOTOX_AREAS:
        max_area_units = max_units.get(area, 30)
        normalized_units = min(injection_number / max_area_units, 1.0)
        effect_strength = 0.35 + (0.3 * (normalized_units ** 0.7))
        strength = min(effect_strength, 0.375)  # Slightly lower max cap for safety

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
        strength = 0.375  # Filler generally uses fixed, subtle strength
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

    # Build negative prompt that protects all other areas
    area_protection = get_protective_negative_prompt(area)
    negative_prompt = f"{common_negative_prompt}, {area_protection}"

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
