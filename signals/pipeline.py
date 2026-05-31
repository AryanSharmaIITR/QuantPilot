"""QuantPilot pipeline orchestrator.

Single entry point for every stage. All behaviour is driven by config.yaml.

Examples
--------
    # Full training pipeline (long history -> features -> train -> evaluate)
    python signals/pipeline.py train-pipeline

    # Daily inference pipeline (short history -> features -> predict)
    python signals/pipeline.py predict-pipeline

    # Individual stages
    python signals/pipeline.py ingest --mode predict
    python signals/pipeline.py preprocess --mode train
    python signals/pipeline.py train
    python signals/pipeline.py predict

Stage reference
---------------
    ingest            Download raw price data (--mode train|predict)
    preprocess        Engineer features        (--mode train|predict)
    train             Train the model (transformers -> NN -> XGBoost) + evaluate
    predict           Emit next-day directional signals
    train-pipeline    ingest(train) -> preprocess(train) -> train
    predict-pipeline  ingest(predict) -> preprocess(predict) -> predict
"""
from __future__ import annotations

import argparse
import sys
import time

from logger import get_logger

log = get_logger("pipeline")


def _ingest(mode: str) -> None:
    from dataIngestion import DataIngestion
    DataIngestion(mode=mode).run()


def _preprocess(mode: str) -> None:
    from dataPreprocessing import Preprocessing
    Preprocessing(mode=mode).run()


def _train() -> None:
    from training import run_training
    run_training()


def _predict() -> None:
    from results import run_prediction
    run_prediction()


def train_pipeline() -> None:
    log.info("########## TRAIN PIPELINE START ##########")
    _ingest("train")
    _preprocess("train")
    _train()
    log.info("########## TRAIN PIPELINE DONE ##########")


def predict_pipeline() -> None:
    log.info("########## PREDICT PIPELINE START ##########")
    _ingest("predict")
    _preprocess("predict")
    _predict()
    log.info("########## PREDICT PIPELINE DONE ##########")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="QuantPilot pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    p_ingest = sub.add_parser("ingest", help="Download raw price data")
    p_ingest.add_argument("--mode", choices=["train", "predict"], default="train")

    p_pre = sub.add_parser("preprocess", help="Engineer features")
    p_pre.add_argument("--mode", choices=["train", "predict"], default="train")

    sub.add_parser("train", help="Train + evaluate the model")
    sub.add_parser("predict", help="Emit next-day directional signals")
    sub.add_parser("train-pipeline", help="ingest+preprocess+train (train mode)")
    sub.add_parser("predict-pipeline", help="ingest+preprocess+predict (predict mode)")

    args = parser.parse_args(argv)

    start = time.monotonic()
    try:
        if args.stage == "ingest":
            _ingest(args.mode)
        elif args.stage == "preprocess":
            _preprocess(args.mode)
        elif args.stage == "train":
            _train()
        elif args.stage == "predict":
            _predict()
        elif args.stage == "train-pipeline":
            train_pipeline()
        elif args.stage == "predict-pipeline":
            predict_pipeline()
    except Exception:
        log.exception("Stage '%s' failed", args.stage)
        return 1

    log.info("Stage '%s' completed in %.1fs", args.stage, time.monotonic() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
