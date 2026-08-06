#!/usr/bin/env python3
"""Resolve the eight scene asset classes without building the full scene."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shelf-usd",
        default=None,
    )
    parser.add_argument(
        "--ball-usd",
        default=None,
    )
    return parser.parse_args()


ARGS = parse_args()

if ARGS.shelf_usd:
    os.environ["SCENEPREDICTOR_SHELF_USD"] = ARGS.shelf_usd
if ARGS.ball_usd:
    os.environ["SCENEPREDICTOR_BALL_USD"] = ARGS.ball_usd


from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
    }
)


from isaac_asset_objects import (
    ASSET_CATALOG,
    resolve_asset_url,
)


def main() -> None:
    try:
        for key in (
            "cereal_box",
            "food_can",
            "bottle",
            "mug",
            "banana",
            "pudding_box",
            "shelf",
            "ball",
        ):
            spec = ASSET_CATALOG[key]
            url = resolve_asset_url(spec)
            print(
                f"{spec.label:<14} {url}",
                flush=True,
            )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
