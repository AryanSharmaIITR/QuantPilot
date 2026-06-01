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


def _ingest(mode: str, end_date: str | None = None) -> None:
    from dataIngestion import DataIngestion
    DataIngestion(mode=mode, end_date=end_date).run()


def _preprocess(mode: str) -> None:
    from dataPreprocessing import Preprocessing
    Preprocessing(mode=mode).run()


def _train() -> None:
    from training import run_training
    run_training()


def _predict(target_date: str | None = None) -> None:
    from results import run_prediction
    run_prediction(target_date)


def train_pipeline() -> None:
    log.info("########## TRAIN PIPELINE START ##########")
    _ingest("train")
    _preprocess("train")
    _train()
    log.info("########## TRAIN PIPELINE DONE ##########")


def predict_pipeline(target_date: str | None = None) -> None:
    log.info("########## PREDICT PIPELINE START ##########")
    # Anchor the download window to the requested date so the prediction is made
    # from one month of history ending the session before it.
    _ingest("predict", end_date=target_date)
    _preprocess("predict")
    _predict(target_date)
    log.info("########## PREDICT PIPELINE DONE ##########")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="QuantPilot pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="stage")

    p_ingest = sub.add_parser("ingest", help="Download raw price data")
    p_ingest.add_argument("--mode", choices=["train", "predict"], default="train")
    p_ingest.add_argument("--as-of-date", dest="as_of_date", default=None,
                          help="Anchor the predict window to end the session before "
                               "this date (ISO). Ignored in train mode.")

    p_pre = sub.add_parser("preprocess", help="Engineer features")
    p_pre.add_argument("--mode", choices=["train", "predict"], default="train")

    sub.add_parser("train", help="Train + evaluate the model")
    p_predict = sub.add_parser("predict", help="Emit directional signals")
    p_predict.add_argument("--as-of-date", dest="as_of_date", default=None,
                           help="Target trading day (ISO). Default: next NSE open.")
    sub.add_parser("train-pipeline", help="ingest+preprocess+train (train mode)")
    p_predict_pipeline = sub.add_parser(
        "predict-pipeline", help="ingest+preprocess+predict (predict mode)")
    p_predict_pipeline.add_argument("--as-of-date", dest="as_of_date", default=None,
                                    help="Target trading day (ISO). Default: next NSE open.")

    args = parser.parse_args(argv)

    # No stage given (e.g. running the file directly from an IDE) — show help.
    if args.stage is None:
        parser.print_help()
        return 0

    start = time.monotonic()
    try:
        if args.stage == "ingest":
            _ingest(args.mode, getattr(args, "as_of_date", None))
        elif args.stage == "preprocess":
            _preprocess(args.mode)
        elif args.stage == "train":
            _train()
        elif args.stage == "predict":
            _predict(getattr(args, "as_of_date", None))
        elif args.stage == "train-pipeline":
            train_pipeline()
        elif args.stage == "predict-pipeline":
            predict_pipeline(getattr(args, "as_of_date", None))
    except Exception:
        log.exception("Stage '%s' failed", args.stage)
        return 1

    log.info("Stage '%s' completed in %.1fs", args.stage, time.monotonic() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
