from __future__ import annotations

from dataclasses import dataclass

from data.data import dev_df, test_df, train_df
from data.pollution import DataPolluter
from models.tfidf_logreg import run_tfidf_logreg
from models.tfidf_lsvm import run_tfidf_lsvm


def print_results(name: str, results: dict, split_name: str) -> None:
    print(f"=== {name} ({split_name}) ===")
    print(f"{split_name} accuracy: {results['accuracy']:.4f}")
    print(results["report"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print()


@dataclass(frozen=True)
class ExperimentConfig:
    feature: str = "description"  # which column to pollute
    pollution_rate: float = 0.0  # fraction in [0,1]
    mode: str = "empty"
    seed: int = 42
    use_feature: str = "both"  # "title" | "description" | "both"


def run_models(
    train_frame, eval_frame, eval_name: str, label: str, use_feature: str
) -> None:
    print_results(
        f"TF-IDF + Logistic Regression [{label} | use={use_feature}]",
        run_tfidf_logreg(train_frame, eval_frame, use_feature=use_feature),
        eval_name,
    )
    print_results(
        f"TF-IDF + Linear SVM [{label} | use={use_feature}]",
        run_tfidf_lsvm(train_frame, eval_frame, use_feature=use_feature),
        eval_name,
    )


def make_polluted_splits(cfg: ExperimentConfig):
    polluter = DataPolluter(seed=cfg.seed)

    if cfg.pollution_rate == 0.0:
        return train_df, dev_df, test_df, "clean"

    train_p = polluter.pollute(
        train_df, cfg.feature, cfg.pollution_rate, mode=cfg.mode
    ).df
    dev_p = polluter.pollute(dev_df, cfg.feature, cfg.pollution_rate, mode=cfg.mode).df
    test_p = polluter.pollute(
        test_df, cfg.feature, cfg.pollution_rate, mode=cfg.mode
    ).df

    label = f"polluted_{cfg.feature}_{int(cfg.pollution_rate * 100)}pct"
    return train_p, dev_p, test_p, label


def main() -> None:
    configs = [
        ExperimentConfig(feature="title", pollution_rate=0.0, use_feature="title"),
        ExperimentConfig(feature="title", pollution_rate=0.1, use_feature="title"),
        ExperimentConfig(feature="title", pollution_rate=0.2, use_feature="title"),
        ExperimentConfig(feature="title", pollution_rate=0.3, use_feature="title"),
        ExperimentConfig(feature="title", pollution_rate=0.4, use_feature="title"),
        ExperimentConfig(feature="title", pollution_rate=0.5, use_feature="title"),
        ExperimentConfig(feature="title", pollution_rate=0.6, use_feature="title"),
        ExperimentConfig(feature="title", pollution_rate=0.7, use_feature="title"),
        ExperimentConfig(feature="title", pollution_rate=0.8, use_feature="title"),
        ExperimentConfig(feature="title", pollution_rate=0.9, use_feature="title"),
    ]

    for cfg in configs:
        tr, dv, te, label = make_polluted_splits(cfg)
        run_models(tr, dv, eval_name="dev", label=label, use_feature=cfg.use_feature)

    RUN_TEST = False
    if RUN_TEST:
        for cfg in configs:
            tr, dv, te, label = make_polluted_splits(cfg)
            run_models(
                tr, te, eval_name="test", label=label, use_feature=cfg.use_feature
            )


if __name__ == "__main__":
    main()
