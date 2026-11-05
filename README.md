# Paper Conservation Analysis

A desktop application for comparing paper images and reporting visible differences between reference and comparison states.

## Overview

This app is designed for a simple conservation workflow:

- Load three images: B1, B2, and A
- Review and adjust masks for paper and ink regions
- Check image alignment visually
- Compare the images using configured analysis options
- View a concise report describing mean and per-pixel differences

## Features

- Load and preview B1, B2, and A images
- Generate paper and ink masks using luminance or reference-colour modes
- Inspect mask overlays and alignment previews
- Compare RGB and LAB image data
- Optionally compare using masks and intersection masks
- Generate a readable analysis summary for each comparison

## Requirements

- Python 3.10+
- PySide6
- NumPy
- SciPy
- Pillow

## Installation

1. Clone the repository.
2. Create and activate a virtual environment if desired.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

If a requirements file is not present, install the packages manually:

```bash
pip install PySide6 numpy scipy pillow
```

## Running the app

From the project root:

```bash
python main.py
```

## Project structure

- `main.py` — desktop app entry point and UI
- `src/paperdiff/` — image loading, analysis, and workflow logic
- `src/paperdiff/image_state.py` — shared image and analysis state objects

## Notes

This project assumes the images have already been aligned and cropped before they are loaded into the application. The app focuses on the comparison and reporting workflow rather than image preparation.
