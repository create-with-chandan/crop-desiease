"""Export the trained Keras model to TensorFlow.js so the scanner runs in the browser.

Only h5py and numpy are needed, so this works without installing TensorFlow:

    python export_tfjs.py --model models/crop_disease.keras --output models/tfjs

Training-only augmentation layers are dropped; they are no-ops at inference time.
Run this again whenever crop_disease.keras and labels.json are replaced.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

SKIPPED_LAYERS = {"Sequential"}
WEIGHT_NAMES = {
    "Conv2D": ["kernel", "bias"],
    "DepthwiseConv2D": ["depthwise_kernel", "bias"],
    "Dense": ["kernel", "bias"],
    "BatchNormalization": ["gamma", "beta", "moving_mean", "moving_variance"],
}


def snake_case(name: str) -> str:
    """Match the Keras naming used for weight groups in model.weights.h5."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()


def layer_config(class_name: str, config: dict) -> dict:
    name = config["name"]
    if class_name == "InputLayer":
        return {"name": name, "batch_input_shape": config["batch_shape"], "dtype": "float32", "sparse": False}
    if class_name == "Rescaling":
        return {"name": name, "scale": config["scale"], "offset": config["offset"]}
    if class_name in {"Conv2D", "DepthwiseConv2D"}:
        result = {
            "name": name,
            "kernel_size": config["kernel_size"],
            "strides": config["strides"],
            "padding": config["padding"],
            "data_format": "channels_last",
            "dilation_rate": config["dilation_rate"],
            "activation": config["activation"],
            "use_bias": config["use_bias"],
        }
        if class_name == "Conv2D":
            result["filters"] = config["filters"]
        else:
            result["depth_multiplier"] = config["depth_multiplier"]
        return result
    if class_name == "BatchNormalization":
        return {
            "name": name,
            "axis": config["axis"],
            "momentum": config["momentum"],
            "epsilon": config["epsilon"],
            "center": config["center"],
            "scale": config["scale"],
        }
    if class_name == "ReLU":
        return {"name": name, "max_value": config.get("max_value")}
    if class_name == "ZeroPadding2D":
        return {"name": name, "padding": config["padding"], "data_format": "channels_last"}
    if class_name == "GlobalAveragePooling2D":
        return {"name": name, "data_format": "channels_last"}
    if class_name == "Dropout":
        return {"name": name, "rate": config["rate"]}
    if class_name == "Dense":
        return {"name": name, "units": config["units"], "activation": config["activation"], "use_bias": config["use_bias"]}
    if class_name == "Add":
        return {"name": name}
    raise ValueError(f"Unsupported layer for TensorFlow.js export: {class_name}")


def history_names(inbound_nodes: list) -> list[str]:
    names: list[str] = []

    def walk(value):
        if isinstance(value, dict):
            if value.get("class_name") == "__keras_tensor__":
                names.append(value["config"]["keras_history"][0])
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for node in inbound_nodes:
        walk(node.get("args", []))
    return names


def export(model_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json"))
        weights_file = h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r")

    layers: list[dict] = []
    weights: list[tuple[str, np.ndarray]] = []
    aliases: dict[str, str] = {}

    def resolve(name: str) -> str:
        while name in aliases:
            name = aliases[name]
        return name

    def add_layers(layer_list: list, weight_group: h5py.Group, outer_inputs: list[str] | None = None) -> None:
        counters: dict[str, int] = defaultdict(int)
        for layer in layer_list:
            class_name = layer["class_name"]
            config = layer["config"]
            base = snake_case(class_name)
            count = counters[base]
            counters[base] += 1
            group_name = base if count == 0 else f"{base}_{count}"

            if class_name == "Functional":
                inner_input = config["input_layers"][0]
                inner_output = config["output_layers"][0]
                aliases[inner_input] = history_names(layer["inbound_nodes"])[0]
                add_layers(config["layers"], weight_group[group_name]["layers"], [inner_input])
                aliases[config["name"]] = inner_output
                continue
            if class_name in SKIPPED_LAYERS:
                aliases[config["name"]] = history_names(layer["inbound_nodes"])[0]
                continue
            if class_name == "InputLayer" and outer_inputs is not None:
                continue

            inbound = [[[resolve(name), 0, 0, {}] for name in history_names(layer["inbound_nodes"])]]
            layers.append(
                {
                    "class_name": class_name,
                    "name": config["name"],
                    "config": layer_config(class_name, config),
                    "inbound_nodes": inbound if inbound[0] else [],
                }
            )

            names = WEIGHT_NAMES.get(class_name)
            if names:
                variables = weight_group[group_name]["vars"]
                if class_name in {"Conv2D", "DepthwiseConv2D", "Dense"} and not config["use_bias"]:
                    names = names[:1]
                if len(variables) != len(names):
                    raise ValueError(f"Weight count mismatch for {config['name']}")
                for index, weight_name in enumerate(names):
                    weights.append((f"{config['name']}/{weight_name}", np.asarray(variables[str(index)], dtype=np.float32)))

    add_layers(config["config"]["layers"], weights_file["layers"])
    weights_file.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_name = "group1-shard1of1.bin"
    with (output_dir / shard_name).open("wb") as shard:
        for _, value in weights:
            shard.write(value.astype("<f4").tobytes())

    model_json = {
        "format": "layers-model",
        "generatedBy": "CropAI export_tfjs.py",
        "convertedBy": None,
        "modelTopology": {
            "class_name": "Model",
            "keras_version": "2.15.0",
            "backend": "tensorflow",
            "config": {
                "name": config["config"]["name"],
                "layers": layers,
                "input_layers": [[resolve(config["config"]["input_layers"][0]), 0, 0]],
                "output_layers": [[resolve(config["config"]["output_layers"][0]), 0, 0]],
            },
        },
        "weightsManifest": [
            {
                "paths": [shard_name],
                "weights": [{"name": name, "shape": list(value.shape), "dtype": "float32"} for name, value in weights],
            }
        ],
    }
    (output_dir / "model.json").write_text(json.dumps(model_json), encoding="utf-8")
    print(f"Exported {len(layers)} layers and {len(weights)} weight tensors to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/crop_disease.keras"))
    parser.add_argument("--output", type=Path, default=Path("models/tfjs"))
    args = parser.parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()
