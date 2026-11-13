"""Visualize PitchOutcomeModel with torchview, without training or pitch data."""

import argparse
import copy
from pathlib import Path

import torch
from torchview import draw_graph

from pitch_model import OUTCOMES, PitchOutcomeModel, load_checkpoint


def draw_architecture(model=None, batch_size=2, direction="TB"):
    """Return a torchview graph for notebook display or Graphviz export.

    Pass an existing PitchOutcomeModel to use its embedding dimensions and
    vocabulary sizes. A copy is traced on CPU, leaving the supplied model's
    weights, device, and train/eval mode untouched. Without a model, use the
    default architecture with illustrative vocabularies of 1,000 players each.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if model is None:
        model = PitchOutcomeModel(num_batters=1000, num_pitchers=1000)
    model = copy.deepcopy(model).cpu().eval()

    # Embeddings and one_hot need integer indices, not random float tensors.
    batter = torch.ones(batch_size, dtype=torch.long)
    pitcher = torch.ones(batch_size, dtype=torch.long)
    # Counts are encoded as 3 * balls + strikes, in the range 0 through 11.
    count = torch.arange(batch_size, dtype=torch.long) % 12

    graph = draw_graph(
        model,
        input_data=(batter, pitcher, count),
        graph_name="PitchOutcomeModel",
        mode="eval",
        device="cpu",
        depth=3,
        expand_nested=True,
        show_shapes=True,
        hide_inner_tensors=True,
        hide_module_functions=True,
        graph_dir=direction,
    )
    graph.visual_graph.attr(
        label=(
            "Pitch outcome model\n"
            "Inputs: batter index, pitcher index, count index (3 * balls + strikes)\n"
            f"Output: {len(OUTCOMES)} logits; softmax is applied separately for prediction"
        ),
        labelloc="t",
        fontsize="16",
    )
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Optional trained .pt checkpoint")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pitch_model_architecture"),
                        help="Output path without a file extension")
    parser.add_argument("--format", nargs="+", choices=("svg", "png", "pdf"),
                        default=["svg", "png"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--direction", choices=("TB", "LR"), default="TB")
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint)[0] if args.checkpoint else None
    graph = draw_architecture(model, args.batch_size, args.direction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for image_format in args.format:
        path = graph.visual_graph.render(
            filename=str(args.output), format=image_format, cleanup=True
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
