"""
SafeVision AI — Main Entry Point
====================================
Unified CLI for the complete PPE detection pipeline.

Usage:
    python main.py prepare              # Generate synthetic training data
    python main.py train                # Train the ML classifier
    python main.py detect               # Run real-time webcam detection
    python main.py detect --source img  # Detect on an image file
"""

import argparse
import sys


def cmd_prepare(args):
    """Generate training data (synthetic or from dataset)."""
    from src.data_prep import generate_synthetic_features, process_image_dataset

    if args.mode == "synthetic":
        generate_synthetic_features(n_samples=args.n_samples)
    else:
        if not args.image_dir:
            print("ERROR: --image-dir required for dataset mode.")
            sys.exit(1)
        process_image_dataset(args.image_dir, args.label_dir)


def cmd_train(args):
    """Train the PPE classifier."""
    from src.train import load_data, train_model, save_model

    X, y = load_data(args.data)
    results = train_model(X, y, model_name=args.algorithm,
                          test_size=args.test_size)
    save_model(results)
    print("\n[OK] Training complete!")


def cmd_detect(args):
    """Run PPE detection inference."""
    from src.inference import PPEInferenceEngine

    engine = PPEInferenceEngine(model_path=args.model,
                                scaler_path=args.scaler)
    source = args.source
    show = not args.no_display

    try:
        source = int(source)
    except ValueError:
        pass

    if isinstance(source, str) and source.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        engine.run_on_image(source, output_path=args.output, show=show)
    else:
        engine.run_on_video(source, output_path=args.output, show=show)


def main():
    parser = argparse.ArgumentParser(
        prog="SafeVision AI",
        description="Real-time PPE Compliance Detection System",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── prepare ─────────────────────────────────────
    p_prep = sub.add_parser("prepare", help="Generate / extract training data")
    p_prep.add_argument("--mode", choices=["synthetic", "dataset"],
                        default="synthetic")
    p_prep.add_argument("--n-samples", type=int, default=2000)
    p_prep.add_argument("--image-dir", type=str, default=None)
    p_prep.add_argument("--label-dir", type=str, default=None)

    # ── train ───────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the PPE classifier")
    p_train.add_argument("--data", type=str, default=None)
    p_train.add_argument("--algorithm",
                         choices=["random_forest", "gradient_boost", "svm"],
                         default="random_forest")
    p_train.add_argument("--test-size", type=float, default=0.2)

    # ── detect ──────────────────────────────────────
    p_det = sub.add_parser("detect", help="Run PPE detection")
    p_det.add_argument("--source", type=str, default="0",
                       help="0=webcam, or path to video/image")
    p_det.add_argument("--output", type=str, default=None)
    p_det.add_argument("--no-display", action="store_true")
    p_det.add_argument("--model", type=str, default=None)
    p_det.add_argument("--scaler", type=str, default=None)

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "detect":
        cmd_detect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
