from PIL import Image
from io import BytesIO
from app.model.sd_model import pipeline

# Base prompts for Botox and Filler treatments (from above)
base_prompts = {
    # BOTOX areas
    "forehead_lines_botox": "Create realistic post-treatment image with smoother horizontal forehead lines proportional to injection units. Natural skin texture, preserved expressions.",
    "frown_lines_glabella_botox": "Show reduction of vertical '11 lines' between eyebrows proportional to injection units. Relaxed muscle appearance, natural look.",
    "crows_feet_botox": "Show softened crow’s feet wrinkles around the eyes proportional to injection units. Natural eye shape and preserved expressions.",
    "nasalis_lines_botox": "Render softened nasal 'bunny' lines proportional to injection units. Maintain natural nose contours.",
    "vertical_lip_lines_botox": "Reduce vertical wrinkles above the lips proportional to injection units. Preserve lip texture and natural contours.",
    "lip_flip_botox": "Enhance the fullness of the upper lip with subtle lift near Cupid’s Bow, proportional to injection units. Maintain natural volume.",
    "smile_lift_botox": "Raise corners of the mouth while reducing downward smile lines proportional to injection units. Preserve natural smile.",
    "masseter_reduction_botox": "Slim jawline and soften lower face appearance by relaxing masseter muscles proportional to injection units. Maintain natural symmetry.",
    "dimpled_chin_botox": "Smooth out dimpled texture of the chin proportional to injection units. Preserve chin shape and definition.",
    "platysmal_bands_botox": "Reduce vertical bands on the neck for a smoother, more youthful neckline proportional to injection units. Maintain natural contours.",
    
    # FILLER areas
    "cheek_filler": "Add volume and lift to the cheeks while maintaining natural facial contours. Balanced, proportionate enhancement.",
    "smile_line_filler": "Fill and soften smile (nasolabial) lines for a youthful, smooth facial appearance while preserving natural expressions.",
    "lip_filler": "Plump and shape the lips with natural, soft volume enhancement while maintaining balanced lip proportions.",
    "temple_filler": "Restore lost volume in the temples for a youthful, lifted facial appearance with natural contours.",
    "nose_filler": "Refine and smooth the nasal bridge and tip with subtle, natural-looking contour improvements."
}


# Define max units for Botox — fillers don't use units the same way
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

# Define Botox and Filler areas separately
BOTOX_AREAS = set(max_units.keys())
FILLER_AREAS = {
    "cheek_filler", "smile_line_filler", "lip_filler", "temple_filler", "nose_filler"
}

def get_ordinal_suffix(n: int) -> str:
    return "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

def generate_images(image: Image.Image, area: str, injection_number: int = 0):
    results = []
    treatment_name = area.replace("_", " ").title()

    prompt = (
        f"Medical photo after "
    )

    if area in BOTOX_AREAS:
        prompt += f"{injection_number} units of Botox in {treatment_name} area. "
        prompt += base_prompts.get(area, "")
        max_area_units = max_units.get(area, 30)
        normalized_units = min(injection_number / max_area_units, 1.0)
        effect_strength = 0.1 + (0.3 * (normalized_units ** 0.7))
        strength = min(effect_strength, 0.35)
    elif area in FILLER_AREAS:
        prompt += f"filler enhancement in {treatment_name} area. "
        prompt += base_prompts.get(area, "")
        strength = 0.3  # fixed strength for filler effects (adjustable)
    else:
        raise ValueError(f"Unknown treatment area: {area}")

    negative_prompt =  "unrealistic, cartoon, distorted features, changed identity"

    print(f"Generating {treatment_name} result{' with ' + str(injection_number) + ' units' if area in BOTOX_AREAS else ''}...")

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
