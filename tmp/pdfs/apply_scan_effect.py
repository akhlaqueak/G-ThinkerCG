from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


SOURCE_IMAGE_DIR = Path(
    "/Users/akahmad/Documents/G-ThinkerCG/tmp/pdfs/g1650-work/pages"
)
PROCESSED_IMAGE_DIR = Path(
    "/Users/akahmad/Documents/G-ThinkerCG/tmp/pdfs/g1650-work/processed"
)
OUTPUT_PDF = Path(
    "/Users/akahmad/Documents/G-ThinkerCG/output/pdf/G-1650_scanned-look-extra.pdf"
)

PAGE_WIDTH_PT = 612
PAGE_HEIGHT_PT = 792


def apply_scan_effect(image: Image.Image, index: int) -> Image.Image:
    gray = image.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(0.96)
    gray = ImageEnhance.Brightness(gray).enhance(0.95)

    array = np.asarray(gray, dtype=np.float32)
    height, width = array.shape
    rng = np.random.default_rng(1729 + index)

    fine_noise = rng.normal(0.0, 4.1, size=array.shape)
    x_gradient = np.linspace(-4.0, 2.5, width, dtype=np.float32)
    y_gradient = np.linspace(2.5, -3.5, height, dtype=np.float32)[:, None]

    yy, xx = np.mgrid[0:height, 0:width]
    distance_from_center = np.sqrt(
        ((xx - width / 2) / (width / 2)) ** 2
        + ((yy - height / 2) / (height / 2)) ** 2
    )
    edge_shading = -10.0 * np.clip(distance_from_center - 0.48, 0, 0.85)

    scanner_band = -3.0 * np.exp(
        -((xx - width * 0.73) ** 2) / (2 * (width * 0.09) ** 2)
    )

    array = (
        array
        + fine_noise
        + x_gradient
        + y_gradient
        + edge_shading
        + scanner_band
    )

    speckle_mask = rng.random(array.shape) < 0.00085
    array[speckle_mask] -= rng.uniform(18, 52, size=speckle_mask.sum())
    array = np.clip(array, 0, 255)

    result = Image.fromarray(array.astype(np.uint8), mode="L")
    result = result.filter(ImageFilter.GaussianBlur(radius=0.36))

    angle = -0.48 if index % 2 == 0 else 0.44
    result = result.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=250,
    )
    return result


def main() -> None:
    PROCESSED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    source_images = sorted(SOURCE_IMAGE_DIR.glob("page-*.png"))
    if not source_images:
        raise RuntimeError("No rendered PDF pages were found.")

    processed_paths = []
    for index, source_path in enumerate(source_images):
        with Image.open(source_path) as image:
            processed = apply_scan_effect(image, index)
            output_image = PROCESSED_IMAGE_DIR / f"page-{index + 1}.jpg"
            processed.save(
                output_image,
                format="JPEG",
                quality=72,
                subsampling=1,
                optimize=True,
                dpi=(300, 300),
            )
            processed_paths.append(output_image)

    pdf = canvas.Canvas(str(OUTPUT_PDF), pagesize=(PAGE_WIDTH_PT, PAGE_HEIGHT_PT))
    pdf.setTitle("G-1650 scanned-look copy")
    for image_path in processed_paths:
        pdf.drawImage(
            ImageReader(str(image_path)),
            0,
            0,
            width=PAGE_WIDTH_PT,
            height=PAGE_HEIGHT_PT,
            preserveAspectRatio=False,
            mask="auto",
        )
        pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
