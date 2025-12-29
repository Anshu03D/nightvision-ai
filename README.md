# Night Haze Removal Project
A computer vision application built on macOS to enhance hazy nighttime images using the Atmospheric Scattering Model.

## ✨ Features
- **Inversion Technique:** Converts night images to "day-like" hazy images for processing.
- **Dark Channel Prior (DCP):** Estimates the haze thickness.
- **Guided Filter:** Refines the transmission map to prevent halos and keep edges sharp.
- **CLAHE Enhancement:** Boosts contrast and brightness in low-light areas.

## 🚀 Installation & Usage
1. Ensure you have Python 3 installed via Homebrew.
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate it: `source .venv/bin/activate`
4. Install dependencies: 
   ```bash
   pip install opencv-python opencv-contrib-python numpy