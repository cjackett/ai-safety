# ruff: noqa: INP001
"""
Generate test images for multimodal safety evaluation.

Creates images containing text prompts for OCR extraction, encoded content,
and cross-modal attack scenarios.
"""

import json
import logging
import textwrap
from pathlib import Path
from typing import Any, cast

import yaml
from PIL import Image, ImageDraw, ImageFont

# Setup module-level logger
logger = logging.getLogger(__name__)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.resolve() / "config" / "config.yaml"

    with config_path.open() as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


def setup_logging() -> None:
    """Setup basic logging for image generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_font_with_fallback(config: dict[str, Any]) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """
    Load font with fallback chain from config.

    Returns:
        ImageFont object
    """
    font_size = config["image_generation"]["font_size"]
    font_chain = config["image_generation"]["font_fallback_chain"]

    for font_name in font_chain:
        try:
            font = ImageFont.truetype(font_name, font_size)
        except OSError:
            logger.debug(f"Font {font_name} not found, trying next...")
        else:
            logger.info(f"Loaded font: {font_name}")
            return font

    logger.warning("No TrueType fonts found, using default PIL font")
    return ImageFont.load_default()


def create_text_image(
    text: str,
    output_path: Path,
    config: dict[str, Any],
    bg_color: str | None = None,
    text_color: str | None = None,
) -> None:
    """Create an image with the specified text."""
    # Get config values
    img_config = config["image_generation"]
    image_size = (img_config["width"], img_config["height"])
    font_size = img_config["font_size"]

    # Use provided colors or defaults from config
    if bg_color is None:
        bg_color = img_config["background_color"]
    if text_color is None:
        text_color = img_config["text_color"]

    # Create image
    img = Image.new("RGB", image_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Load font with fallback
    font = load_font_with_fallback(config)

    # Wrap text to fit image width
    margin = 40
    max_width = image_size[0] - 2 * margin

    # Calculate approximate characters per line
    avg_char_width = font_size * 0.6
    chars_per_line = int(max_width / avg_char_width)

    # Wrap text
    wrapped_lines = []
    for line in text.split("\n"):
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
    logger.debug(f"Generated: {output_path.name}")


def generate_all_images(config: dict[str, Any]) -> None:
    """Generate all test images from prompts JSON."""
    base_dir = Path(__file__).parent.resolve()
    prompts_file = base_dir / config["paths"]["prompts"] / "multimodal_prompts.json"

    logger.info(f"Loading prompts from {prompts_file}")
    with prompts_file.open() as f:
        prompts = json.load(f)

    logger.info("=" * 60)
    logger.info("Generating test images")
    logger.info("=" * 60)

    # Generate OCR injection images
    logger.info("[1/5] OCR Injection Images")
    for prompt in prompts["ocr_injection"]:
        output_path = base_dir / config["paths"]["images"] / "ocr_injection" / f"{prompt['id']}.png"
        create_text_image(prompt["image_text"], output_path, config, bg_color="white", text_color="black")
    logger.info(f"  Generated {len(prompts['ocr_injection'])} OCR injection images")

    # Generate cross-modal images
    logger.info("[2/5] Cross-Modal Inconsistency Images")
    for prompt in prompts["cross_modal"]:
        output_path = base_dir / config["paths"]["images"] / "cross_modal" / f"{prompt['id']}.png"
        create_text_image(prompt["image_text"], output_path, config, bg_color="lightblue", text_color="darkblue")
    logger.info(f"  Generated {len(prompts['cross_modal'])} cross-modal images")

    # Generate encoded harmful images
    logger.info("[3/5] Encoded Harmful Content Images")
    for prompt in prompts["encoded_harmful"]:
        output_path = base_dir / config["paths"]["images"] / "encoded_harmful" / f"{prompt['id']}.png"
        # Show encoded text with label
        display_text = f"[{prompt['attack_type'].upper()}]\n\n{prompt['image_text']}"
        create_text_image(display_text, output_path, config, bg_color="lightyellow", text_color="darkred")
    logger.info(f"  Generated {len(prompts['encoded_harmful'])} encoded harmful images")

    # Generate baseline control images
    logger.info("[4/5] Baseline Control Images")
    for prompt in prompts["baseline_control"]:
        output_path = base_dir / config["paths"]["images"] / "baseline_control" / f"{prompt['id']}.png"
        # Use red background for harmful, green for benign
        bg = "lightcoral" if "harmful" in prompt["attack_type"] else "lightgreen"
        create_text_image(prompt["image_text"], output_path, config, bg_color=bg, text_color="black")
    logger.info(f"  Generated {len(prompts['baseline_control'])} baseline control images")

    # Generate jailbreak roleplay images
    logger.info("[5/5] Jailbreak Roleplay Images")
    for prompt in prompts["jailbreak_roleplay"]:
        output_path = base_dir / config["paths"]["images"] / "ocr_injection" / f"{prompt['id']}.png"
        create_text_image(prompt["image_text"], output_path, config, bg_color="lavender", text_color="darkviolet")
    logger.info(f"  Generated {len(prompts['jailbreak_roleplay'])} jailbreak roleplay images")

    logger.info("=" * 60)
    logger.info("Test image generation complete")
    logger.info("")
    logger.info("Total images generated:")
    logger.info(f"  - OCR Injection: {len(prompts['ocr_injection']) + len(prompts['jailbreak_roleplay'])}")
    logger.info(f"  - Cross-Modal: {len(prompts['cross_modal'])}")
    logger.info(f"  - Encoded Harmful: {len(prompts['encoded_harmful'])}")
    logger.info(f"  - Baseline Control: {len(prompts['baseline_control'])}")
    logger.info(f"\nImages saved to: {base_dir / config['paths']['images']}/")


if __name__ == "__main__":
    setup_logging()
    config = load_config()
    generate_all_images(config)
