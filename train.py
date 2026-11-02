from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CropAI crop disease classifier.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset directory with one folder per class.")
    parser.add_argument("--output", type=Path, default=Path("models"), help="Directory for the model and labels.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--min-images-per-class", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {args.data}")

    class_directories = sorted(path for path in args.data.iterdir() if path.is_dir())
    if len(class_directories) < 2:
        raise SystemExit("The dataset must contain at least two class folders.")
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_counts = {
        path.name: sum(file.suffix.lower() in supported_extensions for file in path.rglob("*"))
        for path in class_directories
    }
    too_small = {name: count for name, count in class_counts.items() if count < args.min_images_per_class}
    if too_small:
        details = ", ".join(f"{name}: {count}" for name, count in too_small.items())
        raise SystemExit(
            f"Each class needs at least {args.min_images_per_class} images for a useful split ({details})."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=args.validation_split,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="categorical",
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=args.validation_split,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise SystemExit("The dataset must contain at least two disease classes.")

    normalization = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = normalization(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    total_images = sum(class_counts.values())
    class_weights = {
        index: total_images / (len(class_names) * class_counts[name])
        for index, name in enumerate(class_names)
    }

    train_ds = train_ds.prefetch(AUTOTUNE)
    validation_ds = validation_ds.prefetch(AUTOTUNE)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "crop_disease.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
    ]
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    model.save(model_path)
    (args.output / "labels.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Saved {model_path}")
    print(f"Saved {args.output / 'labels.json'}")


if __name__ == "__main__":
    main()
