#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DERIVED_TARGET_COL = "Derived_Attack_Class"
DERIVED_TARGET_MODE = "derived_multiclass"
DEFAULT_TARGET_MODE = "column"


def available_column_mask(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    masks = []
    for column in columns:
        if column in df.columns:
            masks.append(df[column].fillna(0).astype(np.float32) > 0)
    if not masks:
        return pd.Series(False, index=df.index)
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined |= mask
    return combined


def derive_multiclass_labels(
    df: pd.DataFrame,
    source_target_col: str = "Attack_label",
) -> pd.Series:
    if source_target_col not in df.columns:
        raise ValueError(
            f"Cannot derive multiclass labels without source target column '{source_target_col}'."
        )

    source_target = pd.to_numeric(df[source_target_col], errors="coerce").fillna(0).astype(int)
    attack_mask = source_target == 1

    labels = pd.Series(
        np.full(len(df), "other_attack", dtype=object),
        index=df.index,
        name=DERIVED_TARGET_COL,
    )
    labels.loc[~attack_mask] = "benign"

    auth_mask = attack_mask & available_column_mask(
        df, ["Login_attempt", "Succesful_login", "is_privileged"]
    )
    labels.loc[auth_mask] = "auth_privilege"

    host_process_mask = attack_mask & ~auth_mask & available_column_mask(
        df, ["File_activity", "Process_activity", "read_write_physical.process"]
    )
    labels.loc[host_process_mask] = "host_process"

    service_series = (
        pd.to_numeric(df["Service"], errors="coerce").fillna(-1)
        if "Service" in df.columns
        else pd.Series(-1, index=df.index)
    )
    port_series = (
        pd.to_numeric(df["Des_port"], errors="coerce").fillna(-1)
        if "Des_port" in df.columns
        else pd.Series(-1, index=df.index)
    )

    web_mask = attack_mask & ~auth_mask & ~host_process_mask & (
        service_series.isin([4, 5]) | port_series.isin([80, 443])
    )
    labels.loc[web_mask] = "web_access"

    dns_mask = attack_mask & ~auth_mask & ~host_process_mask & ~web_mask & (
        service_series.isin([2, 3]) | port_series.isin([53])
    )
    labels.loc[dns_mask] = "dns"

    mqtt_mask = attack_mask & ~auth_mask & ~host_process_mask & ~web_mask & ~dns_mask & (
        service_series.isin([8, 11, 16]) | port_series.isin([1880, 1883])
    )
    labels.loc[mqtt_mask] = "mqtt_iot"

    industrial_mask = (
        attack_mask
        & ~auth_mask
        & ~host_process_mask
        & ~web_mask
        & ~dns_mask
        & ~mqtt_mask
        & (service_series.isin([0, 7, 12, 15]) | port_series.isin([502, 4321, 5683]))
    )
    labels.loc[industrial_mask] = "industrial_control"

    return labels


def apply_target_mode(
    df: pd.DataFrame,
    target_col: str,
    target_mode: str,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    working = df.copy()
    if target_mode == DERIVED_TARGET_MODE:
        working[DERIVED_TARGET_COL] = derive_multiclass_labels(
            working,
            source_target_col=target_col,
        )
        return working, DERIVED_TARGET_COL, target_col, [DERIVED_TARGET_COL, target_col]

    if target_col not in working.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(working.columns)}"
        )
    return working, target_col, target_col, [target_col]


def prepare_dataframe_for_model(
    df: pd.DataFrame,
    target_mode: str,
    target_col: str,
    source_target_col: str,
    drop_columns: list[str],
) -> tuple[pd.DataFrame, str]:
    working = df.copy()
    if target_mode == DERIVED_TARGET_MODE and target_col not in working.columns:
        if source_target_col not in working.columns:
            raise ValueError(
                f"Cannot prepare multiclass stream data without source target column "
                f"'{source_target_col}'."
            )
        working[target_col] = derive_multiclass_labels(
            working,
            source_target_col=source_target_col,
        )
    return working, target_col


def humanize_label(label: str) -> str:
    text = str(label)
    if text == "1":
        return "Attack"
    if text == "0":
        return "Benign"
    mapping = {
        "benign": "Benign",
        "auth_privilege": "Auth Privilege",
        "host_process": "Host Process",
        "web_access": "Web Access",
        "dns": "DNS",
        "mqtt_iot": "MQTT IoT",
        "industrial_control": "Industrial Control",
        "other_attack": "Other Attack",
    }
    if text in mapping:
        return mapping[text]
    return text.replace("_", " ").replace("/", " / ").title()
