"""
Generate test images for multimodal safety evaluation.

Creates images containing text prompts for OCR extraction, encoded content,
and cross-modal attack scenarios.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Script directory
SCRIPT_DIR = Path(__file__).parent


def create_text_image(
    text: str,
    output_path: Path,
    image_size: tuple = (800, 600),
    font_size: int = 24,
    bg_color: str = "white",
    text_color: str = "black",
) -> None:
    """Create an image with the specified text."""

    # Create image
    img = Image.new('RGB', image_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a system font, fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except OSError:
            # Fall back to default font
            font = ImageFont.load_default()

    # Wrap text to fit image width
    margin = 40
    max_width = image_size[0] - 2 * margin

    # Calculate approximate characters per line
    avg_char_width = font_size * 0.6
    chars_per_line = int(max_width / avg_char_width)

    # Wrap text
    wrapped_lines = []
    for line in text.split('\n'):
        wrapped_lines.extend(textwrap.wrap(line, width=chars_per_line))

    # Draw text
    y_offset = margin
    line_height = font_size + 10

    for line in wrapped_lines:
        draw.text((margin, y_offset), line, fill=text_color, font=font)
        y_offset += line_height

    # Save image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"  ✓ Generated: {output_path.name}")


def generate_all_test_images():
    """Generate all test images from prompts JSON."""

    prompts_file = SCRIPT_DIR / "prompts" / "multimodal_prompts.json"

    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    print("\nGenerating test images...")
    print("="*60)

    # Generate OCR injection images
    print("\n[1/5] OCR Injection Images")
    for prompt in prompts["ocr_injection"]:
        output_path = SCRIPT_DIR / "test_images" / "ocr_injection" / f"{prompt['id']}.png"
        create_text_image(
            prompt["image_text"],
            output_path,
            bg_color="white",
            text_color="black"
        )

    # Generate cross-modal images
    print("\n[2/5] Cross-Modal Inconsistency Images")
    for prompt in prompts["cross_modal"]:
        output_path = SCRIPT_DIR / "test_images" / "cross_modal" / f"{prompt['id']}.png"
        create_text_image(
            prompt["image_text"],
            output_path,
            bg_color="lightblue",
            text_color="darkblue"
        )

    # Generate encoded harmful images
    print("\n[3/5] Encoded Harmful Content Images")
    for prompt in prompts["encoded_harmful"]:
        output_path = SCRIPT_DIR / "test_images" / "encoded_harmful" / f"{prompt['id']}.png"
        # Show encoded text with label
        display_text = f"[{prompt['attack_type'].upper()}]\n\n{prompt['image_text']}"
        create_text_image(
            display_text,
            output_path,
            bg_color="lightyellow",
            text_color="darkred"
        )

    # Generate baseline control images
    print("\n[4/5] Baseline Control Images")
    for prompt in prompts["baseline_control"]:
        output_path = SCRIPT_DIR / "test_images" / "baseline_control" / f"{prompt['id']}.png"
        # Use red background for harmful, green for benign
        bg = "lightcoral" if "harmful" in prompt["attack_type"] else "lightgreen"
        create_text_image(
            prompt["image_text"],
            output_path,
            bg_color=bg,
            text_color="black"
        )

    # Generate jailbreak roleplay images
    print("\n[5/5] Jailbreak Roleplay Images")
    for prompt in prompts["jailbreak_roleplay"]:
        output_path = SCRIPT_DIR / "test_images" / "ocr_injection" / f"{prompt['id']}.png"
        create_text_image(
            prompt["image_text"],
            output_path,
            bg_color="lavender",
            text_color="darkviolet"
        )

    print("\n" + "="*60)
    print("✓ Test image generation complete!")
    print(f"\nTotal images generated:")
    print(f"  - OCR Injection: {len(prompts['ocr_injection']) + len(prompts['jailbreak_roleplay'])}")
    print(f"  - Cross-Modal: {len(prompts['cross_modal'])}")
    print(f"  - Encoded Harmful: {len(prompts['encoded_harmful'])}")
    print(f"  - Baseline Control: {len(prompts['baseline_control'])}")
    print(f"\nImages saved to: {SCRIPT_DIR / 'test_images'}/")


if __name__ == "__main__":
    generate_all_test_images()
