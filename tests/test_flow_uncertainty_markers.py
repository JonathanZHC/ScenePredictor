"""Flow-uncertainty region geometry and the /scene_predictor/flow_uncertainty_markers MarkerArray.

Needs the ROS message packages (source /opt/ros/<distro>/setup.bash); no ROS graph
or rclpy.init is used: the node and publishers are fakes.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("rclpy")
pytest.importorskip("visualization_msgs.msg")

from builtin_interfaces.msg import Time  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

from scene_pred_pipeline import ros_output  # noqa: E402
from scene_pred_pipeline.config import (  # noqa: E402
    DifFlowConfig,
    DifFlowUncertaintyCalibrationConfig,
    DifFlowUncertaintyHeadConfig,
    OutputConfig,
    PipelineConfig,
    RosConfig,
)
from scene_pred_pipeline.data_types import SceneVelocityOutput  # noqa: E402
from scene_pred_pipeline.ros_output import (  # noqa: E402
    _UNCERTAINTY_COLOR_RADIUS_M,
    _UNCERTAINTY_RAMP,
    RosVisualizer,
    ellipsoid_marker_geometry,
    uncertainty_colors,
    uncertainty_region_markers,
)


def _quaternion_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        -2,
    )


def _random_spd(rng, count, scale_m=0.01):
    a = rng.normal(size=(count, 3, 3)) * scale_m
    return a @ np.swapaxes(a, 1, 2) + np.eye(3) * 1.0e-8


# ---------------------------------------------------------------- geometry
@pytest.mark.parametrize("factor", [1.0, 2.0, 0.5])
def test_ellipsoid_geometry_rebuilds_scaled_covariance(factor):
    rng = np.random.default_rng(0)
    sigma = _random_spd(rng, 500)
    c = rng.uniform(0.5, 8.0, size=500)
    covs = (factor * c)[:, None, None] ** 2 * sigma
    centers = rng.normal(size=(500, 3))
    # Some raw eigenvector frames are reflections; those must come out right-handed too.
    assert np.any(np.linalg.det(np.linalg.eigh(covs)[1]) < 0.0)

    positions, quaternions, diameters = ellipsoid_marker_geometry(centers, covs)

    assert positions.shape == (500, 3) and np.array_equal(positions, centers)
    assert quaternions.shape == (500, 4) and diameters.shape == (500, 3)
    assert np.allclose(np.linalg.norm(quaternions, axis=1), 1.0)
    rotations = _quaternion_matrix(quaternions)
    # Proper (right-handed) rotations: orthonormal with determinant +1.
    assert np.allclose(rotations @ np.swapaxes(rotations, 1, 2), np.eye(3), atol=1.0e-9)
    assert np.allclose(np.linalg.det(rotations), 1.0)
    rebuilt = rotations @ (((diameters / 2.0) ** 2)[:, :, None] * np.swapaxes(rotations, 1, 2))
    assert np.allclose(rebuilt, covs, rtol=1.0e-7, atol=1.0e-12 * factor**2)


def test_ellipsoid_geometry_axis_aligned_and_empty():
    cov = np.diag([0.04, 0.01, 0.0025])[None]
    _, quaternions, diameters = ellipsoid_marker_geometry(np.zeros((1, 3)), cov)
    rotation = _quaternion_matrix(quaternions)[0]
    # Every local axis is a world axis; diameter = 2 sqrt(variance) along it.
    assert np.allclose(np.abs(rotation), np.abs(rotation).round())
    assert np.allclose(sorted(diameters[0]), [0.1, 0.2, 0.4])
    assert np.allclose(rotation @ np.diag((diameters[0] / 2) ** 2) @ rotation.T, cov[0])

    positions, quaternions, diameters = ellipsoid_marker_geometry(np.zeros((0, 3)), np.zeros((0, 3, 3)))
    assert positions.shape == (0, 3) and quaternions.shape == (0, 4) and diameters.shape == (0, 3)


def test_uncertainty_ramp_is_ordered_and_not_red():
    colors = uncertainty_colors(np.geomspace(0.001, 1.0, 64))
    assert colors.shape == (64, 3) and np.all((colors >= 0.0) & (colors <= 1.0))
    # Monotone brightness (perceptually ordered); green always dominates over a
    # red-only hue so nothing reads as the CBF critical-point red.
    luminance = colors @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(luminance) >= -1.0e-12)
    assert np.all(colors[:, 1] >= 0.35)


def test_uncertainty_ramp_ends_at_2_and_30_mm():
    assert _UNCERTAINTY_COLOR_RADIUS_M == (0.002, 0.03)
    colors = uncertainty_colors(np.array([0.001, 0.002, 0.03, 0.1, np.nan]))
    assert np.allclose(colors[:2], _UNCERTAINTY_RAMP[0])
    # NaN reads as maximal uncertainty.
    assert np.allclose(colors[2:], _UNCERTAINTY_RAMP[-1])


def test_region_markers_ellipsoid_and_ball():
    rng = np.random.default_rng(1)
    sigma = _random_spd(rng, 50)
    s = rng.uniform(0.5, 8.0, size=50)
    centers = rng.normal(size=(50, 3))
    factor = 2.0
    radius = s * np.linalg.det(sigma) ** (1.0 / 6.0)

    positions, quaternions, diameters, one_frame = uncertainty_region_markers(
        "ellipsoid", centers, sigma, s, factor
    )
    assert np.array_equal(positions, centers) and np.allclose(one_frame, radius)
    rotations = _quaternion_matrix(quaternions)
    rebuilt = rotations @ (((diameters / 2.0) ** 2)[:, :, None] * np.swapaxes(rotations, 1, 2))
    assert np.allclose(rebuilt, (factor * s)[:, None, None] ** 2 * sigma, rtol=1.0e-7, atol=1.0e-14)
    # Equal-volume radius of the drawn ellipsoid = factor * the one-frame radius.
    assert np.allclose(np.prod(diameters / 2.0, axis=1) ** (1.0 / 3.0), factor * radius)

    positions, quaternions, diameters, one_frame = uncertainty_region_markers(
        "ball", centers, sigma, s, factor
    )
    assert np.array_equal(positions, centers) and np.allclose(one_frame, radius)
    assert np.array_equal(quaternions, np.tile([0.0, 0.0, 0.0, 1.0], (50, 1)))
    assert np.allclose(diameters, 2.0 * factor * radius[:, None])

    with pytest.raises(ValueError, match="region"):
        uncertainty_region_markers("sphere", centers, sigma, s, factor)


# ---------------------------------------------------------------- MarkerArray
class _Publisher:
    def __init__(self, topic):
        self.topic = topic
        self.messages = []

    def get_subscription_count(self):
        return 1

    def publish(self, message):
        self.messages.append(message)


class _Node:
    def __init__(self):
        self.publishers = {}

    def create_publisher(self, message_type, topic, qos):
        publisher = _Publisher(topic)
        self.publishers[topic] = publisher
        return publisher

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time()))


_TOPIC = "/scene_predictor/flow_uncertainty_markers"


def _visualizer(region="ellipsoid", head_enabled=True, **output):
    values = dict(
        publish_tracked_objects=False,
        publish_rest_scene=False,
        publish_velocity_cloud=False,
        publish_velocity_markers=True,
        publish_tracked_masks=False,
        publish_annotated_rgb=False,
        publish_flow_anchors=False,
        visualization_publish_hz=0.0,
        velocity_marker_stride=2,
        velocity_marker_scale_factor=2.0,
        publish_uncertainty_markers=True,
        uncertainty_marker_alpha=0.35,
    )
    values.update(output)
    head = DifFlowUncertaintyHeadConfig(
        enable=head_enabled,
        checkpoint="x",
        calibration=DifFlowUncertaintyCalibrationConfig(region=region),
    )
    config = PipelineConfig(
        ros=RosConfig(world_frame="world_test", camera_names=("camera_0",)),
        output=OutputConfig(**values),
        difflow=DifFlowConfig(uncertainty_head=head),
    )
    node = _Node()
    return RosVisualizer(node, config), node


def _output(device, *, with_cov=True, anchor_device=None):
    """Two tracks: the first half of a line along x (track 1, y = 0) and the second half
    moved to y = 0.5 (track 2).

    Decoy track-2 anchors at y = 0.01 are closer to the track-1 dense points than the
    track-1 anchors (y = 0.05), so a cross-track nearest neighbour would pick the wrong
    covariance. The anchor tensors (positions, ids, covariance, scale) live on
    anchor_device (default: device).
    """
    anchor_device = device if anchor_device is None else anchor_device
    n = 400
    x = torch.linspace(0.0, 1.0, n, device=device)
    points = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    points[n // 2:, 1] = 0.5
    track_ids = torch.ones(n, dtype=torch.int32, device=device)
    track_ids[n // 2:] = 2
    velocity = torch.zeros_like(points)
    velocity[:, 2] = 0.3
    velocity[n // 2:, 0] = -0.6

    # Anchors: track 1 at y = 0.05 (far), track 2 copies at y = 0.01 (near) of the
    # track-1 points, plus the real track-2 anchors at y = 0.5.
    a1 = points[: n // 2: 4].clone()
    a1[:, 1] = 0.05
    decoy = points[: n // 2: 4].clone()
    decoy[:, 1] = 0.01
    a2 = points[n // 2:: 4].clone()
    anchors = torch.cat((a1, decoy, a2)).to(anchor_device)
    anchor_ids = torch.cat(
        (
            torch.ones(a1.shape[0], dtype=torch.int64),
            torch.full((decoy.shape[0] + a2.shape[0],), 2, dtype=torch.int64),
        )
    ).to(anchor_device)
    sigma1 = torch.diag(torch.tensor([4.0e-6, 1.0e-6, 2.5e-7]))
    sigma2 = torch.diag(torch.tensor([9.0e-6, 9.0e-6, 9.0e-6]))
    cov = torch.cat(
        (sigma1.expand(a1.shape[0], 3, 3), sigma2.expand(decoy.shape[0] + a2.shape[0], 3, 3))
    ).to(anchor_device)
    scale = torch.cat(
        (torch.full((a1.shape[0],), 2.0), torch.full((decoy.shape[0] + a2.shape[0],), 3.0))
    ).to(anchor_device)
    empty = torch.empty((0, 3), device=device)
    return SceneVelocityOutput(
        stamp_ns=1_500_000_000,
        flow_dt_s=0.05,
        tracked_points=points,
        tracked_track_ids=track_ids,
        flow_points=points,
        flow_velocity=velocity,
        flow_track_ids=track_ids,
        source_anchors=anchors,
        warped_anchors=anchors,
        removed_outlier_points=empty,
        view_results={},
        common_track_ids=(1, 2),
        flow_valid=True,
        timings_ms={},
        anchor_track_ids=anchor_ids,
        anchor_flow_cov=cov if with_cov else None,
        anchor_flow_cov_scale=scale if with_cov else None,
    )


_DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
# (flow_points device, anchor tensors device), including mixed placements.
_PLACEMENTS = [(device, device) for device in _DEVICES] + (
    [("cuda:0", "cpu"), ("cpu", "cuda:0")] if torch.cuda.is_available() else []
)
_SIGMA1 = np.diag([4.0e-6, 1.0e-6, 2.5e-7])
_SIGMA2 = np.diag([9.0e-6] * 3)


def _marker_covariances(markers) -> np.ndarray:
    rotations = _quaternion_matrix(
        np.array([[m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w] for m in markers])
    )
    diameters = np.array([[m.scale.x, m.scale.y, m.scale.z] for m in markers])
    return rotations @ (((diameters / 2.0) ** 2)[:, :, None] * np.swapaxes(rotations, 1, 2))


@pytest.mark.parametrize("device, anchor_device", _PLACEMENTS)
def test_uncertainty_marker_array(device, anchor_device):
    visualizer, node = _visualizer(uncertainty_marker_max=40)
    output = _output(device, anchor_device=anchor_device)
    visualizer.publish(output)

    (array,) = node.publishers[_TOPIC].messages
    markers = array.markers
    assert markers[0].action == Marker.DELETEALL
    ellipsoids = markers[1:]
    # 400 points, stride 2 -> 200 drawn lines, capped at 40 ellipsoids.
    assert len(ellipsoids) == 40
    # RViz 2 rejects the whole array if two markers (DELETEALL included) share ns and id.
    assert len({(m.ns, m.id) for m in markers}) == len(markers)
    for marker in markers:
        assert marker.header.frame_id == "world_test"
        assert marker.header.stamp.sec == 1 and marker.header.stamp.nanosec == 500_000_000
    assert {m.ns for m in markers} == {"flow_uncertainty"}
    assert all(m.type == Marker.SPHERE and m.action == Marker.ADD for m in ellipsoids)
    assert all(m.color.a == pytest.approx(0.35) for m in ellipsoids)

    # Evenly spread over the lines: the first and last drawn line are included.
    centers = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z] for m in ellipsoids])
    points = output.flow_points.cpu().numpy()[::2]
    velocity = output.flow_velocity.cpu().numpy()[::2]
    tips = points + 0.05 * 2.0 * velocity
    assert np.allclose(centers[0], tips[0], atol=1e-6) and np.allclose(centers[-1], tips[-1], atol=1e-6)

    # Each tip carries the covariance of its SAME-track nearest anchor (the decoys
    # of track 2 sit closer to track 1), scaled by factor^2 s^2 (region ellipsoid).
    covs = _marker_covariances(ellipsoids)
    track_one = centers[:, 1] < 0.25
    assert 0 < track_one.sum() < len(ellipsoids)
    assert np.allclose(covs[track_one], (2.0 * 2.0) ** 2 * _SIGMA1, atol=1e-9)
    assert np.allclose(covs[~track_one], (2.0 * 3.0) ** 2 * _SIGMA2, atol=1e-9)
    # Colour follows the one-frame radius s det(Sigma)^(1/6), independent of factor.
    radius_one = 2.0 * np.linalg.det(_SIGMA1) ** (1.0 / 6.0)
    radius_two = 3.0 * np.linalg.det(_SIGMA2) ** (1.0 / 6.0)
    colors = np.array([[m.color.r, m.color.g, m.color.b] for m in ellipsoids])
    assert np.allclose(colors[track_one], uncertainty_colors(np.array([radius_one])), atol=1e-6)
    assert np.allclose(colors[~track_one], uncertainty_colors(np.array([radius_two])), atol=1e-6)

    # The velocity LINE_LIST is unchanged by the shared line extraction.
    (velocity_array,) = node.publishers["/scene_predictor/velocity_markers"].messages
    line_list = velocity_array.markers[1]
    assert len(line_list.points) == 2 * 200
    assert line_list.points[1].z == pytest.approx(tips[0, 2])


def test_ball_region_draws_spheres():
    visualizer, node = _visualizer(region="ball", uncertainty_marker_max=40)
    output = _output("cpu")
    visualizer.publish(output)
    (array,) = node.publishers[_TOPIC].messages
    assert array.markers[0].action == Marker.DELETEALL
    spheres = array.markers[1:]
    assert len(spheres) == 40 and all(m.type == Marker.SPHERE for m in spheres)
    # Sphere of radius factor * s * det(Sigma)^(1/6) at the line tip, unrotated.
    radius_one = 2.0 * np.linalg.det(_SIGMA1) ** (1.0 / 6.0)
    radius_two = 3.0 * np.linalg.det(_SIGMA2) ** (1.0 / 6.0)
    centers = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z] for m in spheres])
    diameters = np.array([[m.scale.x, m.scale.y, m.scale.z] for m in spheres])
    track_one = centers[:, 1] < 0.25
    assert 0 < track_one.sum() < len(spheres)
    assert np.allclose(diameters[track_one], 2.0 * 2.0 * radius_one)
    assert np.allclose(diameters[~track_one], 2.0 * 2.0 * radius_two)
    assert all(
        (m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w)
        == (0.0, 0.0, 0.0, 1.0)
        for m in spheres
    )
    colors = np.array([[m.color.r, m.color.g, m.color.b] for m in spheres])
    assert np.allclose(colors[~track_one], uncertainty_colors(np.array([radius_two])), atol=1e-6)


def test_default_cap_is_150():
    assert OutputConfig().uncertainty_marker_max == 150
    visualizer, node = _visualizer(velocity_marker_stride=1)
    visualizer.publish(_output("cpu"))
    (array,) = node.publishers[_TOPIC].messages
    # 400 drawn lines, 150 ellipsoids spread over them (both ends included).
    assert len(array.markers) == 1 + 150


class _Clock:
    def __init__(self):
        self.now_ns = 1_000_000_000

    def __call__(self):
        return self.now_ns


def _published(node):
    return (
        len(node.publishers["/scene_predictor/velocity_markers"].messages),
        len(node.publishers[_TOPIC].messages),
    )


def test_uncertainty_rate_gate(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(ros_output.time, "monotonic_ns", clock)
    # Every frame passes the global gate (0 = unthrottled); ellipsoids at 5 Hz.
    visualizer, node = _visualizer(visualization_publish_hz=0.0, uncertainty_marker_publish_hz=5.0)
    output = _output("cpu")
    counts = []
    for _ in range(7):                       # frames 0.1 s apart
        visualizer.publish(output)
        counts.append(_published(node))
        clock.now_ns += 100_000_000
    assert [velocity for velocity, _ in counts] == [1, 2, 3, 4, 5, 6, 7]
    assert [uncertainty for _, uncertainty in counts] == [1, 1, 2, 2, 3, 3, 4]
    # Every uncertainty publish starts with DELETEALL.
    assert all(a.markers[0].action == Marker.DELETEALL for a in node.publishers[_TOPIC].messages)


def test_uncertainty_gate_is_on_top_of_the_global_gate(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(ros_output.time, "monotonic_ns", clock)
    # Global gate 4 Hz (250 ms) is slower than the ellipsoid gate (10 Hz): it decides, and the
    # ellipsoids go out with every global publish.
    visualizer, node = _visualizer(visualization_publish_hz=4.0, uncertainty_marker_publish_hz=10.0)
    output = _output("cpu")
    counts = []
    for _ in range(6):                       # frames 0.1 s apart: phase-locked global gate passes at 0, 0.3, 0.5 s
        visualizer.publish(output)
        counts.append(_published(node))
        clock.now_ns += 100_000_000
    assert counts == [(1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (3, 3)]


def test_rate_gate_keeps_the_configured_average_rate():
    # 27 Hz frames through a 10 Hz gate for 10 s: 100 publishes (now + period scheduling: 90).
    next_ns, period_ns, due_count = 0, 100_000_000, 0
    for frame in range(270):
        due, next_ns = ros_output._rate_gate(next_ns, period_ns, 1_000_000_000 + frame * 37_037_037)
        due_count += int(due)
    assert due_count == 100
    # After a pause longer than one period: one publish, then the rate again (no burst).
    now = 1_000_000_000 + 270 * 37_037_037 + 2_000_000_000
    dues = []
    for frame in range(6):
        due, next_ns = ros_output._rate_gate(next_ns, period_ns, now + frame * 37_037_037)
        dues.append(due)
    assert dues == [True, False, False, True, False, False]


def test_uncertainty_markers_on_the_velocity_line_frames(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(ros_output.time, "monotonic_ns", clock)
    # Equal rates (fr3_real: 10 / 10): the ellipsoids are published with every velocity-line publish.
    visualizer, node = _visualizer(visualization_publish_hz=10.0, uncertainty_marker_publish_hz=10.0)
    output = _output("cpu")
    counts = []
    for _ in range(81):                      # 27 Hz frames for 3 s
        visualizer.publish(output)
        counts.append(_published(node))
        clock.now_ns += 37_037_037
    assert all(velocity == uncertainty for velocity, uncertainty in counts)
    assert counts[-1][0] == 30


def test_uncertainty_marker_pool_is_reused_without_stale_markers():
    # Gates open (every frame, ellipsoids with every publish): three publishes in a row.
    visualizer, node = _visualizer(
        uncertainty_marker_max=40, visualization_publish_hz=0.0, uncertainty_marker_publish_hz=1.0e9
    )
    visualizer.publish(_output("cpu"))
    visualizer.publish(_output("cpu", with_cov=False))
    visualizer.publish(_output("cpu"))
    first, cleared, third = node.publishers[_TOPIC].messages
    assert len(first.markers) == 41 and len(cleared.markers) == 1 and len(third.markers) == 41
    # Same Marker objects, ids 1..40 next to the (flow_uncertainty, 0) DELETEALL.
    assert all(a is b for a, b in zip(first.markers[1:], third.markers[1:]))
    assert sorted(m.id for m in third.markers[1:]) == list(range(1, 41))
    assert len({(m.ns, m.id) for m in third.markers}) == len(third.markers)


def test_rate_summary_counts_real_publishes(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(ros_output.time, "monotonic_ns", clock)
    visualizer, _ = _visualizer(visualization_publish_hz=10.0, uncertainty_marker_publish_hz=10.0)
    output = _output("cpu")
    for _ in range(27):
        visualizer.publish(output)
        clock.now_ns += 37_037_037
    text = visualizer.rate_summary()
    assert "velocity_markers" in text and "flow_uncertainty_markers" in text
    assert visualizer.rate_summary().endswith("none")       # counters restart after each summary


def test_uncertainty_markers_only_clear_without_covariance():
    visualizer, node = _visualizer(publish_velocity_markers=False)
    visualizer.publish(_output("cpu", with_cov=False))
    (array,) = node.publishers[_TOPIC].messages
    assert len(array.markers) == 1 and array.markers[0].action == Marker.DELETEALL
    assert array.markers[0].header.frame_id == "world_test"


def test_uncertainty_marker_cap_and_small_frames():
    visualizer, node = _visualizer(uncertainty_marker_max=500, velocity_marker_stride=1)
    output = _output("cpu")
    visualizer.publish(output)
    (array,) = node.publishers[_TOPIC].messages
    # Fewer lines than the cap: one ellipsoid per drawn line.
    assert len(array.markers) == 1 + 400

    visualizer, node = _visualizer(uncertainty_marker_max=1)
    visualizer.publish(output)
    (array,) = node.publishers[_TOPIC].messages
    assert len(array.markers) == 2


def test_uncertainty_publisher_absent_without_head():
    # On by default (OutputConfig), but no head means no covariance and no topic.
    assert OutputConfig().publish_uncertainty_markers
    visualizer, node = _visualizer(head_enabled=False, publish_uncertainty_markers=True)
    assert visualizer.uncertainty_marker_pub is None and _TOPIC not in node.publishers
    visualizer.publish(_output("cpu"))
    assert "/scene_predictor/velocity_markers" in node.publishers


def test_uncertainty_publisher_absent_when_disabled():
    visualizer, node = _visualizer(publish_uncertainty_markers=False)
    assert visualizer.uncertainty_marker_pub is None and _TOPIC not in node.publishers
    assert not visualizer._uncertainty_markers_due()
    visualizer.publish(_output("cpu"))
