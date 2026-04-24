# -*- coding: utf-8 -*-
"""
Optuna-based hyperparameter optimisation (HPO) utilities for deep forecasting models.

Protocol (fair scientific evaluation)
--------------------------------------
- HPO stage  : each trial trains ONLY on a train/val split of train_valid_data
               (controlled by ``hpo_cfg["train_ratio_in_tv"] < 1``).
               Objective = ``model.best_val_loss`` exposed by
               :class:`~ts_benchmark.baselines.deep_forecasting_model_base.DeepForecastingModelBase`.
- Refit stage: caller is responsible for refitting a fresh model with the returned
               ``best_params`` on the full ``train_valid_data`` (``train_ratio_in_tv=1``).
- Test  stage: handled entirely by the calling strategy; no test data is accessed here.
"""
import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_PARAM_TYPES = {"float", "int", "categorical"}


def _suggest_param(trial: Any, name: str, spec: Dict) -> Any:
    """
    Map a generic JSON-schema-like spec to an Optuna ``suggest_*`` call.

    Supported spec shapes
    ---------------------
    - ``{"type": "float",       "low": ..., "high": ..., "log": true/false}``
    - ``{"type": "int",         "low": ..., "high": ..., "log": true/false}``
    - ``{"type": "categorical", "choices": [...]}``

    :param trial: An Optuna ``Trial`` object.
    :param name: Parameter name (used as the Optuna parameter name).
    :param spec: Parameter specification dict.
    :return: The sampled value.
    :raises ValueError: If ``spec["type"]`` is not one of the supported types.
    """
    param_type = spec.get("type")
    if param_type == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    elif param_type == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    elif param_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(
            f"Unknown search space param type: {param_type!r}. "
            f"Supported types: {_SUPPORTED_PARAM_TYPES}"
        )


def run_optuna_hpo(
    model_factory: Any,
    train_valid_data: Any,
    covariates: Optional[Dict],
    hpo_cfg: Dict,
    seed: Optional[int],
    deterministic_mode: Optional[str],
) -> Tuple[Dict, float]:
    """
    Run Optuna HPO over ``train_valid_data`` and return the best hyperparameters.

    Each trial:
    1. Samples hyperparameters from ``hpo_cfg["search_space"]``.
    2. Creates a **fresh** model instance by merging the factory's base hyperparameters
       with the sampled ones.
    3. Calls ``model.forecast_fit(train_valid_data, train_ratio_in_tv=hpo_cfg["train_ratio_in_tv"])``,
       which internally creates a validation split.
    4. Returns ``model.best_val_loss`` as the Optuna objective (minimised).

    Test data is **never** accessed inside this function.

    :param model_factory: A :class:`~ts_benchmark.models.ModelFactory` instance whose
        ``model_factory`` callable and ``model_hyper_params`` are used to create models.
    :param train_valid_data: Training+validation DataFrame.  Must NOT contain test data.
    :param covariates: Covariates dict (e.g. ``{"exog": ...}``).
    :param hpo_cfg: HPO configuration dict with at minimum the keys:
        - ``n_trials``        (int)
        - ``train_ratio_in_tv`` (float < 1)
        - ``search_space``    (dict of param specs)
        - ``timeout_sec``     (int, optional)
    :param seed: Random seed for reproducibility.
    :param deterministic_mode: ``"full"`` or ``"efficient"``; aligns with
        :meth:`~ts_benchmark.evaluation.strategy.forecasting.ForecastingStrategy.execute`.
    :return: A tuple ``(best_params, best_value)`` where ``best_params`` is a dict of
        the best suggested hyperparameters and ``best_value`` is the corresponding
        validation loss.
    :raises ImportError: If ``optuna`` is not installed.
    :raises ValueError: If ``hpo_cfg["train_ratio_in_tv"] >= 1`` or
        ``hpo_cfg["n_trials"] < 1``.
    """
    try:
        import optuna  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "optuna is required for HPO but is not installed. "
            "Install it with:  pip install optuna"
        ) from exc

    from ts_benchmark.utils.random_utils import (  # noqa: PLC0415
        fix_all_random_seed,
        fix_random_seed,
    )

    n_trials: int = hpo_cfg["n_trials"]
    timeout_sec: Optional[int] = hpo_cfg.get("timeout_sec", None)
    hpo_train_ratio: float = hpo_cfg["train_ratio_in_tv"]
    search_space: Dict = hpo_cfg.get("search_space", {})

    if hpo_train_ratio >= 1.0:
        raise ValueError(
            f"hpo.train_ratio_in_tv must be < 1, got {hpo_train_ratio}"
        )
    if n_trials < 1:
        raise ValueError(f"hpo.n_trials must be >= 1, got {n_trials}")

    base_params = deepcopy(model_factory.model_hyper_params)

    def objective(trial: Any) -> float:
        # Set seed per-trial for reproducibility while keeping trials independent.
        trial_seed = (seed + trial.number) if seed is not None else None
        if deterministic_mode == "full":
            fix_all_random_seed(trial_seed)
        elif deterministic_mode == "efficient":
            fix_random_seed(trial_seed)

        # Sample hyperparameters from the search space.
        trial_params = {
            name: _suggest_param(trial, name, spec)
            for name, spec in search_space.items()
        }

        # Merge base params with suggested params (suggested params take priority).
        params = {**base_params, **trial_params}

        # Create a brand-new model instance for this trial.
        model = model_factory.model_factory(**params)

        # Train on the HPO train/val split (train_ratio_in_tv < 1 creates a val set).
        fit_method = (
            model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        )
        fit_method(
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=hpo_train_ratio,
        )

        val_loss = float(getattr(model, "best_val_loss", np.inf))
        return val_loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    best_params: Dict = study.best_params
    best_value: float = study.best_value

    logger.info(
        "HPO completed: best_value=%.6f, best_params=%s", best_value, best_params
    )

    return best_params, best_value
