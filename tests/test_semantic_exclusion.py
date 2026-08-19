from __future__ import annotations

import time
import unittest

import numpy as np

from scene_pred_pipeline.config import _tracker_config
from scene_pred_pipeline.profiler import CycleProfiler
from sam_rgbd_tracking.batched_postprocess import BatchedPostprocessor, _MaskRecord
from sam_rgbd_tracking.component import SAMTrackingComponent
from sam_rgbd_tracking.config import Config
from sam_rgbd_tracking.data_types import (
    CameraIntrinsics,
    DetectionInstance,
    RGBDFrame,
    TrackState,
)
from sam_rgbd_tracking.slots import SlotSpec, max_cross_frame_instances


class PromptConfigTests(unittest.TestCase):
    def test_tracked_and_excluded_are_validated(self) -> None:
        cfg = _tracker_config(
            {
                "tracked_prompts": [["human", 2]],
                "excluded_prompts": [["robot", 1]],
            },
            base_dir=__import__("pathlib").Path("."),
        )
        self.assertEqual(cfg.tracked_prompts, (("human", 2),))
        self.assertEqual(cfg.excluded_prompts, (("robot", 1),))

        with self.assertRaisesRegex(ValueError, "both tracked and excluded"):
            _tracker_config(
                {
                    "tracked_prompts": [["robot", 1]],
                    "excluded_prompts": [["robot", 1]],
                },
                base_dir=__import__("pathlib").Path("."),
            )

    def test_excluded_capacity_does_not_size_3d_alignment(self) -> None:
        native = Config(
            {
                "runtime": {"camera_names": ["camera_0", "camera_1"]},
                "detector": {
                    "prompts": [["human", 2], ["robot", 5]],
                    "excluded_labels": ["robot"],
                },
            }
        )
        # Worst-case cross-view grouping bound: views * tracked capacity only.
        self.assertEqual(max_cross_frame_instances(native), 4)


class SemanticMorphologyTests(unittest.TestCase):
    def test_excluded_mask_dilates_and_never_enters_tracked_pending(self) -> None:
        config = Config(
            {
                "runtime": {"device": "cpu", "gpu_postprocess": False},
                "detector": {"excluded_labels": ["robot"]},
                "postprocess": {
                    "mask_threshold": 0.0,
                    "tracking_erosion_pixels": 1,
                    "exclusion_dilation_pixels": 1,
                    "gpu_batch": False,
                    "gpu_geometry": False,
                    "cpu_workers": 1,
                },
            }
        )
        post = BatchedPostprocessor(config, 1)
        try:
            records = [
                _MaskRecord(0, 0, 1, excluded=False),
                _MaskRecord(0, 1, 2, excluded=True),
            ]
            logits = np.full((2, 5, 5), -1.0, dtype=np.float32)
            logits[0, 1:4, 1:4] = 1.0  # tracked 3x3 -> eroded 1 pixel
            logits[1, 2, 2] = 1.0      # excluded 1 pixel -> dilated 3x3

            pending, exclusion = post._batch_masks_cpu(
                records,
                {(5, 5, 5, 5): [0, 1]},
                [logits],
                need_raw_masks=True,
                need_final_masks=True,
            )

            self.assertEqual([item[0].track_id for item in pending], [1])
            self.assertEqual(int(np.asarray(pending[0][1], dtype=bool).sum()), 1)
            self.assertIsNotNone(exclusion[0])
            self.assertEqual(int(np.asarray(exclusion[0], dtype=bool).sum()), 9)
        finally:
            post.close()


class SlotContinuityTests(unittest.TestCase):
    @staticmethod
    def _frame() -> RGBDFrame:
        return RGBDFrame(
            camera_name="camera_0",
            frame_index=10,
            timestamp_ns=10,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_m=np.ones((4, 4), dtype=np.float32),
            intrinsics=CameraIntrinsics(1.0, 1.0, 0.0, 0.0, 4, 4),
        )

    @staticmethod
    def _component(label: str, *, excluded: bool) -> SAMTrackingComponent:
        component = object.__new__(SAMTrackingComponent)
        component.local_slot_iou_threshold = 0.05
        component.excluded_labels = frozenset({label}) if excluded else frozenset()
        component.slot_layout = [
            SlotSpec(0, 1, label, 0),
            SlotSpec(1, 2, label, 1),
        ]
        empty = np.zeros((4, 4), dtype=bool)
        component.tracks = {
            1: TrackState(
                track_id=1, label=label, semantic_confidence=1.0,
                class_slot=0, active=True, last_mask=empty.copy(),
                last_raw_mask=empty.copy(),
            ),
            2: TrackState(
                track_id=2, label=label, semantic_confidence=0.0,
                class_slot=1, active=False, last_mask=empty.copy(),
                last_raw_mask=empty.copy(),
            ),
        }
        return component

    def test_tracked_class_keeps_original_empty_fallback_slot_semantics(self) -> None:
        component = self._component("human", excluded=False)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        detection = DetectionInstance(0, "human", 0.9, mask)
        masks, activated = component.build_direct_correction_masks(
            self._frame(), [detection], [1, 2], {1: np.zeros((4, 4), bool)}
        )
        self.assertEqual(activated, 1)
        self.assertTrue(component.tracks[1].active)
        self.assertTrue(component.tracks[2].active)
        self.assertFalse(np.any(masks[0]))
        self.assertTrue(np.array_equal(masks[1], mask))

    def test_excluded_class_can_reuse_active_empty_fallback_slot(self) -> None:
        component = self._component("robot", excluded=True)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        detection = DetectionInstance(0, "robot", 0.9, mask)
        masks, activated = component.build_direct_correction_masks(
            self._frame(), [detection], [1, 2], {1: np.zeros((4, 4), bool)}
        )
        self.assertEqual(activated, 0)
        self.assertTrue(component.tracks[1].active)
        self.assertFalse(component.tracks[2].active)
        self.assertTrue(np.array_equal(masks[0], mask))



class ProfilerTests(unittest.TestCase):
    def test_per_frame_decomposition_is_additive(self) -> None:
        profiler = CycleProfiler(enabled=False)
        profiler.start_cycle()

        with profiler.stage("tracker_total", cuda=False):
            time.sleep(0.001)
        tracker_total = profiler._cpu_values["tracker_total"]
        profiler.record("tracking_model", tracker_total * 0.45)
        profiler.record("postprocess", tracker_total * 0.20)
        profiler.record("alignment", tracker_total * 0.15)
        profiler.record("adapter", tracker_total * 0.10)

        with profiler.stage("instance_filter", cuda=False):
            pass
        with profiler.stage("difflow_total", cuda=False):
            with profiler.stage("source_prepare", cuda=False):
                pass
            with profiler.stage("target_prepare", cuda=False):
                pass
            with profiler.stage("inference", cuda=False):
                time.sleep(0.001)
            with profiler.stage("output_extract", cuda=False):
                pass
        with profiler.stage("velocity_recovery", cuda=False):
            pass

        values = profiler.finish()
        self.assertAlmostEqual(
            values["tracker_total"],
            values["tracking_model"]
            + values["postprocess"]
            + values["alignment"]
            + values["adapter"]
            + values["tracker_other"],
            places=9,
        )
        self.assertAlmostEqual(
            values["difflow_total"],
            values["source_prepare"]
            + values["target_prepare"]
            + values["inference"]
            + values["output_extract"]
            + values["difflow_other"],
            places=9,
        )
        self.assertAlmostEqual(
            values["cycle_total"],
            values["tracker_total"]
            + values["instance_filter"]
            + values["difflow_total"]
            + values["velocity_recovery"]
            + values["cycle_other"],
            places=9,
        )
        summary = profiler.format_summary()
        self.assertIn("End-to-end numerical cycle", summary)
        self.assertIn("difflow_total", summary)
        self.assertNotIn("tracker/", summary)


if __name__ == "__main__":
    unittest.main()
