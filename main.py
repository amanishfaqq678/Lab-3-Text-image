"""
LAB 03: Transform Textual and Image Data into Machine-Understandable Format
Run: python main.py
Based on the supplied Lab 03 PDF/DOCX and notebook.
"""
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer


def text_feature_engineering():
    print("\n" + "=" * 70)
    print("PART A: TEXT FEATURE ENGINEERING (BAG OF WORDS)")
    print("=" * 70)

    documents = [
        "The sun is shining today.",
        "The weather is good, the sun is great.",
        "A sunny day is a wonderful day."
    ]

    print("\n--- Raw Text Documents ---")
    for i, doc in enumerate(documents, start=1):
        print(f"Doc {i}: {doc}")

    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    text_matrix = X_text.toarray()

    print("\n--- Bag-of-Words (BoW) Transformation ---")
    print(f"Vocabulary (Features): {feature_names}")
    print(f"Shape of Feature Matrix: {text_matrix.shape}")

    df_text = pd.DataFrame(
        text_matrix,
        columns=feature_names,
        index=[f"Doc {i}" for i in range(1, len(documents) + 1)]
    )
    print("\nBoW Numerical Feature Matrix:")
    print(df_text.to_string())

    print("\n--- CountVectorizer Defaults ---")
    print(f"Lowercase enabled: {vectorizer.lowercase}")
    print(f"Token pattern: {vectorizer.token_pattern}")
    print("'a' included in vocabulary:", "a" in feature_names)
    print("'the' included in vocabulary:", "the" in feature_names)

    total_cells = text_matrix.size
    zero_cells = int((text_matrix == 0).sum())
    print("\n--- Matrix Sparsity ---")
    print(f"Matrix cells: {total_cells}")
    print(f"Zero cells: {zero_cells}")
    print(f"Sparsity: {zero_cells / total_cells:.1%}")

    # Exercise 1: unigrams compared with unigrams + bigrams
    print("\n--- Exercise 1: Unigrams and Bigrams ---")
    uni = CountVectorizer(ngram_range=(1, 1))
    uni_matrix = uni.fit_transform(documents)
    uni_bi = CountVectorizer(ngram_range=(1, 2))
    uni_bi_matrix = uni_bi.fit_transform(documents)
    bigrams = [term for term in uni_bi.get_feature_names_out() if " " in term]
    print(f"ngram_range=(1,1): {uni_matrix.shape[1]} features")
    print(f"ngram_range=(1,2): {uni_bi_matrix.shape[1]} features")
    print(f"New bigram features: {len(bigrams)}")
    print("Bigrams:", bigrams)

    # Exercise 2: add a fourth document with repeated 'sun'
    print("\n--- Exercise 2: Add a Fourth Document ---")
    documents_4 = documents + ["sun sun sun sun sun and more sun today"]
    vec4 = CountVectorizer()
    matrix4 = vec4.fit_transform(documents_4).toarray()
    df4 = pd.DataFrame(
        matrix4,
        columns=vec4.get_feature_names_out(),
        index=[f"Doc {i}" for i in range(1, 5)]
    )
    print(df4.to_string())
    print("\nTotal token counts per document:")
    print(df4.sum(axis=1).to_string())
    print("Doc 4 contains repeated occurrences of 'sun'; raw counts increase with repetition.")


def image_to_features(path, size=(64, 64)):
    """Load an image, resize, convert to grayscale, and return a 1-D pixel vector.

    size is (width, height). The returned vector length is width * height.
    """
    with Image.open(path) as source:
        img = source.convert("RGB")
        img = img.resize(size)
        img = img.convert("L")
        return np.asarray(img, dtype=np.uint8).flatten()


def image_feature_engineering():
    print("\n" + "=" * 70)
    print("PART B: IMAGE FEATURE ENGINEERING")
    print("=" * 70)

    image_file_path = None  # Keep None to use the self-contained dummy image.

    if image_file_path and os.path.exists(image_file_path):
        try:
            original_image = Image.open(image_file_path).convert("RGB")
            print(f"--- Loaded Image from File: {image_file_path} ---")
        except Exception as exc:
            print(f"Could not load image: {exc}")
            original_image = create_dummy_image()
    else:
        print("No valid image file path provided or file not found.")
        original_image = create_dummy_image()

    original_image_array = np.asarray(original_image)
    h, w, c = original_image_array.shape
    print(f"--- Created/Prepared Image ({w}x{h} RGB) ---")
    print(f"Original Image Size (H, W, Channels): {original_image_array.shape}")
    print(f"Total Features (Pixels) in RGB: {h * w * c}")

    grayscale_image = original_image.convert("L")
    grayscale_array = np.asarray(grayscale_image)
    print("\n--- Grayscale Conversion (Feature Reduction) ---")
    print(f"Grayscale Image Size (H, W): {grayscale_array.shape}")
    print(f"Total Features (Pixels) after Grayscale: {grayscale_array.size}")

    flattened_features = grayscale_array.flatten()
    print("\n--- Pixel Flattening (Vectorization) ---")
    print(f"Flattened Feature Vector Shape: {flattened_features.shape}")
    print("\nFirst 10 Features (Pixel Values):")
    print(flattened_features[:10])
    print(f"\nData Type of Final Features: {flattened_features.dtype}")

    # Exercise 3: demonstrate resizing to two fixed dimensions.
    print("\n--- Exercise 3: Resizing Demonstration ---")
    for target in [(64, 64), (32, 32)]:
        resized_gray = original_image.resize(target).convert("L")
        vector = np.asarray(resized_gray).flatten()
        print(f"Resized to {target[0]}x{target[1]} -> vector shape {vector.shape}")

    # Exercise 4: verify same vector length for differently sized images.
    print("\n--- Exercise 4: image_to_features() Verification ---")
    os.makedirs("test_images", exist_ok=True)
    dimensions = [(120, 80), (300, 300), (64, 200)]
    for i, (width, height) in enumerate(dimensions, start=1):
        pixels = np.random.default_rng(i).integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        Image.fromarray(pixels, "RGB").save(f"test_images/sample_{i}.png")

    for i in range(1, 4):
        features = image_to_features(f"test_images/sample_{i}.png", size=(64, 64))
        print(f"Image {i}: {features.shape} (expected (4096,))")


def create_dummy_image():
    """Create a reproducible 100x100 RGB image for the lab demonstration."""
    rng = np.random.default_rng(42)
    dummy_data = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    return Image.fromarray(dummy_data, "RGB")


def home_assignment():
    print("\n" + "=" * 70)
    print("HOME ASSIGNMENT DATASET CHECK")
    print("=" * 70)
    print("Place 10 or more images in each folder:")
    print("  data/class_a/")
    print("  data/class_b/")
    print("The script will create dataset.npz when images are available.")

    class_dirs = {0: "data/class_a", 1: "data/class_b"}
    X_list, y_list = [], []

    for label, folder in class_dirs.items():
        paths = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            paths.extend(glob(os.path.join(folder, pattern)))
        for path in sorted(paths):
            try:
                X_list.append(image_to_features(path, size=(64, 64)))
                y_list.append(label)
            except Exception as exc:
                print(f"[SKIP] {path}: {exc}")

    if X_list:
        X = np.vstack(X_list)
        y = np.asarray(y_list)
        np.savez("dataset.npz", X=X, y=y)
        print(f"X shape: {X.shape}; y shape: {y.shape}")
        print(f"Samples: {X.shape[0]}; Features per sample: {X.shape[1]}")
        print("Saved dataset.npz")
    else:
        print("No assignment images found. Main Lab 03 practical is still complete.")
        print("For the 20-image assignment, add 10 images to each class folder.")


if __name__ == "__main__":
    text_feature_engineering()
    image_feature_engineering()
    home_assignment()
    print("\n" + "=" * 70)
    print("LAB 03 COMPLETED SUCCESSFULLY")
    print("=" * 70)
