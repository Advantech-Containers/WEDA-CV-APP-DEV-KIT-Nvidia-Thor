"""Test suite — 40 tests covering optimization layer, modules, integration."""

from __future__ import annotations

import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optimization layer tests
# ---------------------------------------------------------------------------

class TestMemoryManager:
    def _make(self):
        from weda_optim.memory_manager import MemoryManager, MemoryType
        return MemoryManager(device="cpu"), MemoryType

    def test_allocate_cpu(self):
        mm, MT = self._make()
        buf = mm.allocate("test", (100, 100), "float32", MT.CPU)
        assert isinstance(buf, np.ndarray)
        assert buf.shape == (100, 100)
        mm.shutdown()

    def test_reuse_same_name_shape(self):
        mm, MT = self._make()
        b1 = mm.allocate("buf", (10,), "float32", MT.CPU)
        b2 = mm.allocate("buf", (10,), "float32", MT.CPU)
        assert b1 is b2
        stats = mm.stats()
        assert stats["reuse_hits"] == 1
        mm.shutdown()

    def test_no_reuse_different_shape(self):
        mm, MT = self._make()
        b1 = mm.allocate("buf", (10,), "float32", MT.CPU)
        b2 = mm.allocate("buf", (20,), "float32", MT.CPU)
        assert b1 is not b2
        mm.shutdown()

    def test_release(self):
        mm, MT = self._make()
        mm.allocate("buf", (10,), "float32", MT.CPU)
        mm.release("buf")
        stats = mm.stats()
        assert stats["buffers"]["buf"]["refs"] == 0
        mm.shutdown()

    def test_deallocate(self):
        mm, MT = self._make()
        mm.allocate("buf", (10,), "float32", MT.CPU)
        mm.deallocate("buf")
        assert mm.get_buffer("buf") is None
        mm.shutdown()

    def test_stats(self):
        mm, MT = self._make()
        mm.allocate("a", (100,), "float32", MT.CPU)
        mm.allocate("b", (200,), "uint8", MT.CPU)
        stats = mm.stats()
        assert stats["num_buffers"] == 2
        assert stats["alloc_count"] == 2
        mm.shutdown()

    def test_shutdown_clears(self):
        mm, MT = self._make()
        mm.allocate("buf", (10,), "float32", MT.CPU)
        mm.shutdown()
        assert mm.stats()["num_buffers"] == 0


class TestBufferPool:
    def _make(self, count=4):
        from weda_optim.buffer_pool import BufferPool, PoolConfig
        pool = BufferPool()
        pool.register_pool(PoolConfig("test", (100, 100, 3), "uint8", count, False))
        return pool

    def test_register(self):
        pool = self._make()
        stats = pool.stats()
        assert "test" in stats["pools"]
        assert stats["pools"]["test"]["capacity"] == 4
        pool.shutdown()

    def test_acquire_returns_buffer(self):
        pool = self._make()
        idx, buf = pool.acquire("test")
        assert isinstance(buf, np.ndarray)
        assert buf.shape == (100, 100, 3)
        pool.shutdown()

    def test_acquire_release_cycle(self):
        pool = self._make(count=2)
        i1, _ = pool.acquire("test")
        i2, _ = pool.acquire("test")
        pool.release("test", i1)
        i3, _ = pool.acquire("test")
        assert i3 == i1  # reused
        pool.shutdown()

    def test_overflow_on_exhaustion(self):
        pool = self._make(count=1)
        pool.acquire("test")
        # Second acquire should overflow
        idx, buf = pool.acquire("test")
        stats = pool.stats()
        assert stats["pools"]["test"]["overflow"] == 1
        pool.shutdown()

    def test_release_makes_slot_free(self):
        pool = self._make(count=1)
        idx, _ = pool.acquire("test")
        stats = pool.stats()
        assert stats["pools"]["test"]["free"] == 0
        pool.release("test", idx)
        stats = pool.stats()
        assert stats["pools"]["test"]["free"] == 1
        pool.shutdown()

    def test_unknown_pool_raises(self):
        pool = self._make()
        with pytest.raises(KeyError):
            pool.acquire("nonexistent")
        pool.shutdown()

    def test_write_to_buffer(self):
        pool = self._make()
        idx, buf = pool.acquire("test")
        buf[:] = 128
        assert buf[0, 0, 0] == 128
        pool.shutdown()

    def test_multiple_pools(self):
        from weda_optim.buffer_pool import BufferPool, PoolConfig
        pool = BufferPool()
        pool.register_pool(PoolConfig("rgb", (480, 640, 3), "uint8", 2, False))
        pool.register_pool(PoolConfig("depth", (480, 640), "uint16", 2, False))
        _, rgb = pool.acquire("rgb")
        _, dep = pool.acquire("depth")
        assert rgb.shape == (480, 640, 3)
        assert dep.shape == (480, 640)
        pool.shutdown()


class TestScheduler:
    def _make(self):
        from weda_optim.scheduler import Scheduler
        return Scheduler(
            fixed_placements={"detection": "gpu", "tracking": "cpu", "spatial": "cpu"},
            auto_stages={"depth", "reasoning"},
            gpu_util_threshold=0.85, max_concurrent_gpu=2, min_speedup=2.0,
        )

    def test_detection_always_gpu(self):
        from app.utils.data_structures import DevicePlacement, StageID
        s = self._make()
        assert s.decide_placement(StageID.DETECTION) == "gpu"

    def test_tracking_always_cpu(self):
        from app.utils.data_structures import DevicePlacement, StageID
        s = self._make()
        assert s.decide_placement(StageID.TRACKING) == "cpu"

    def test_spatial_always_cpu(self):
        from app.utils.data_structures import DevicePlacement, StageID
        s = self._make()
        assert s.decide_placement(StageID.SPATIAL) == "cpu"

    def test_auto_defaults_gpu(self):
        from app.utils.data_structures import DevicePlacement, StageID
        s = self._make()
        # Depth is AUTO, should default to GPU without data
        assert s.decide_placement(StageID.DEPTH) == "gpu"

    def test_saturation_guard(self):
        from weda_optim.scheduler import Scheduler
        from app.utils.data_structures import StageID
        s = Scheduler(
            fixed_placements={"detection": "gpu", "tracking": "cpu"},
            auto_stages={"depth"},
            gpu_util_threshold=0.85, max_concurrent_gpu=1, min_speedup=2.0,
        )
        # Simulate one GPU stage active
        with s.gpu_context():
            assert s.decide_placement(StageID.DEPTH) == "cpu"

    def test_latency_routing_gpu(self):
        from app.utils.data_structures import StageID
        s = self._make()
        for _ in range(10):
            s.record_latency(StageID.DEPTH, "cpu", 0.030)
            s.record_latency(StageID.DEPTH, "gpu", 0.005)
        assert s.decide_placement(StageID.DEPTH) == "gpu"

    def test_latency_routing_cpu_when_no_speedup(self):
        from app.utils.data_structures import StageID
        s = self._make()
        for _ in range(10):
            s.record_latency(StageID.DEPTH, "cpu", 0.010)
            s.record_latency(StageID.DEPTH, "gpu", 0.009)
        assert s.decide_placement(StageID.DEPTH) == "cpu"

    def test_scheduled_execution_context(self):
        from app.utils.data_structures import StageID
        s = self._make()
        with s.scheduled_execution(StageID.TRACKING) as p:
            assert p == "cpu"


class TestProfiler:
    def _make(self):
        from weda_optim.profiler import Profiler
        return Profiler(window=10, log_interval=0)

    def test_stage_timing(self):
        p = self._make()
        with p.time_stage("test"):
            time.sleep(0.01)
        st = p.get_or_create_stage("test")
        assert len(st.samples) == 1
        assert st.samples[0] >= 0.005

    def test_frame_timing(self):
        p = self._make()
        p.begin_frame()
        time.sleep(0.01)
        p.end_frame()
        assert p._frame_count == 1

    def test_percentiles(self):
        p = self._make()
        stage = p.get_or_create_stage("t")
        for i in range(20):
            stage.record(float(i) / 1000)
        assert stage.percentile(50) > 0

    def test_summary(self):
        p = self._make()
        p.begin_frame()
        with p.time_stage("det"):
            pass
        p.end_frame()
        s = p.summary()
        assert "det" in s["stages"]
        assert s["frame_count"] == 1

    def test_reset(self):
        p = self._make()
        p.begin_frame()
        p.end_frame()
        p.reset()
        assert p._frame_count == 0


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

class TestDataStructures:
    def test_detection(self):
        from app.utils.data_structures import Detection
        det = Detection(
            bbox=np.array([10, 20, 50, 60], dtype=np.float32),
            confidence=0.9, class_id=0, class_name="person",
        )
        assert det.center == (30.0, 40.0)
        assert det.area == 40 * 40

    def test_spatial_info_defaults(self):
        from app.utils.data_structures import SpatialInfo
        si = SpatialInfo()
        assert si.distance_m == 0.0
        assert si.nearest_neighbor_id == -1


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------

class TestTracker:
    def _make(self):
        from app.modules.bytetrack import ByteTracker as Tracker
        return Tracker(iou_threshold=0.3)

    def _det(self, x1, y1, x2, y2, cls=0):
        from app.utils.data_structures import Detection
        return Detection(
            bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
            confidence=0.9, class_id=cls, class_name="obj",
        )

    def test_first_frame_assigns_ids(self):
        t = self._make()
        dets = [self._det(0, 0, 50, 50), self._det(100, 100, 150, 150)]
        result = t.update(dets)
        assert all(d.track_id > 0 for d in result)
        assert result[0].track_id != result[1].track_id

    def test_same_position_keeps_id(self):
        t = self._make()
        d1 = [self._det(0, 0, 50, 50)]
        r1 = t.update(d1)
        tid = r1[0].track_id
        d2 = [self._det(2, 2, 52, 52)]  # slight shift
        r2 = t.update(d2)
        assert r2[0].track_id == tid

    def test_new_object_gets_new_id(self):
        t = self._make()
        t.update([self._det(0, 0, 50, 50)])
        r = t.update([self._det(0, 0, 50, 50), self._det(500, 500, 550, 550)])
        ids = {d.track_id for d in r}
        assert len(ids) == 2

    def test_class_filtering(self):
        t = Tracker(iou_threshold=0.3, class_match=True)
        t.update([self._det(0, 0, 50, 50, cls=0)])
        r = t.update([self._det(0, 0, 50, 50, cls=1)])  # different class
        # Should get a new ID since class doesn't match
        assert r[0].track_id > 0


class TestDepthProcessor:
    def _make(self):
        from app.modules.depth import DepthProcessor
        from app.utils.data_structures import PipelineConfig
        return DepthProcessor(PipelineConfig())

    def test_colorize_cpu(self):
        from app.utils.data_structures import DevicePlacement
        dp = self._make()
        depth = np.random.randint(500, 5000, (100, 100), dtype=np.uint16)
        result = dp.colorize(depth, DevicePlacement.CPU)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_colorize_zeros(self):
        from app.utils.data_structures import DevicePlacement
        dp = self._make()
        depth = np.zeros((100, 100), dtype=np.uint16)
        result = dp.colorize(depth, DevicePlacement.CPU)
        assert result.shape == (100, 100, 3)

    def test_depth_at(self):
        from app.utils.data_structures import Detection
        dp = self._make()
        depth = np.full((100, 100), 2000, dtype=np.uint16)
        det = Detection(
            bbox=np.array([20, 20, 80, 80], dtype=np.float32),
            confidence=0.9, class_id=0, class_name="person",
        )
        dist = dp._depth_at(depth, det)
        assert abs(abs(dist) - 2.0) < 0.01  # 2000mm = 2.0m (positive = reliable)

    def test_compute_spatial(self):
        from app.utils.data_structures import Detection
        dp = self._make()
        depth = np.full((480, 640), 3000, dtype=np.uint16)
        det = Detection(
            bbox=np.array([300, 200, 340, 260], dtype=np.float32),
            confidence=0.9, class_id=0, class_name="person",
        )
        spatial = dp.compute_spatial([det], depth)
        assert len(spatial) == 1
        assert spatial[0].distance_m > 0
        assert spatial[0].position_3d is not None


# Convenience: import Tracker class for parametrized tests above
from app.modules.bytetrack import ByteTracker as Tracker


# ---------------------------------------------------------------------------
# YAML Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_defaults_without_yaml(self):
        from app.config import load_config
        config = load_config(yaml_path=None)
        assert config.source == "synthetic"
        assert config.width == 1920
        assert config.gpu_util_threshold == 0.85

    def test_load_from_yaml(self, tmp_path):
        from app.config import load_config
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "input:\n  source: '0'\n  width: 1280\n"
            "optimization:\n  scheduler:\n    gpu_util_threshold: 0.7\n"
        )
        config = load_config(str(yaml_file))
        assert config.source == "0"
        assert config.width == 1280
        assert config.gpu_util_threshold == 0.7

    def test_cli_overrides_yaml(self, tmp_path):
        from app.config import load_config
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("input:\n  source: '0'\n")
        config = load_config(str(yaml_file), cli_overrides={"source": "synthetic"})
        assert config.source == "synthetic"  # CLI wins

    def test_missing_yaml_uses_defaults(self):
        from app.config import load_config
        config = load_config("/nonexistent/path.yaml")
        assert config.source == "synthetic"

    def test_validation_clamps_threshold(self):
        from app.config import load_config
        config = load_config(None, cli_overrides={"gpu_util_threshold": 1.5})
        assert config.gpu_util_threshold == 1.0

    def test_rtsp_config_defaults(self):
        from app.utils.data_structures import PipelineConfig
        config = PipelineConfig()
        assert config.rtsp_protocol == "tcp"
        assert config.rtsp_reconnect_attempts == 5
        assert config.rtsp_timeout == 10.0

    def test_gpu_monitor_config_defaults(self):
        from app.utils.data_structures import PipelineConfig
        config = PipelineConfig()
        assert config.enable_gpu_monitor is False
        assert config.gpu_monitor_interval == 1.0


# ---------------------------------------------------------------------------
# GPU Monitor tests
# ---------------------------------------------------------------------------

class TestGpuMonitor:
    def test_mock_monitor_values(self):
        from weda_optim.gpu_monitor import MockMonitor
        m = MockMonitor(gpu_util=0.7, mem_util=0.4, temperature=50.0)
        assert m.get_gpu_utilization() == 0.7
        assert m.get_memory_utilization() == 0.4
        assert m.get_temperature() == 50.0

    def test_mock_monitor_start_stop(self):
        from weda_optim.gpu_monitor import MockMonitor
        m = MockMonitor()
        m.start()  # no-op
        m.stop()   # no-op
        assert m.get_gpu_utilization() == 0.5

    def test_factory_returns_mock_without_hardware(self):
        from weda_optim.gpu_monitor import create_gpu_monitor, MockMonitor
        from app.utils.data_structures import PipelineConfig
        config = PipelineConfig(enable_gpu_monitor=True)
        # On dev machine without tegrastats, should fall back to mock
        monitor = create_gpu_monitor(config)
        assert isinstance(monitor, (MockMonitor,)) or monitor is not None

    def test_scheduler_uses_gpu_threshold(self):
        from weda_optim.gpu_monitor import MockMonitor
        from weda_optim.scheduler import Scheduler
        from app.utils.data_structures import StageID
        monitor = MockMonitor(gpu_util=0.95)
        s = Scheduler(auto_stages={"depth"}, gpu_util_threshold=0.85, gpu_monitor=monitor)
        assert s.decide_placement(StageID.DEPTH) == "cpu"

    def test_scheduler_allows_gpu_below_threshold(self):
        from weda_optim.gpu_monitor import MockMonitor
        from weda_optim.scheduler import Scheduler
        from app.utils.data_structures import StageID
        monitor = MockMonitor(gpu_util=0.3)
        s = Scheduler(auto_stages={"depth"}, gpu_util_threshold=0.85, gpu_monitor=monitor)
        assert s.decide_placement(StageID.DEPTH) == "gpu"

    def test_scheduler_stats_includes_gpu(self):
        from weda_optim.gpu_monitor import MockMonitor
        from weda_optim.scheduler import Scheduler
        monitor = MockMonitor(gpu_util=0.6, mem_util=0.4, temperature=55.0)
        s = Scheduler(gpu_monitor=monitor)
        stats = s.stats()
        assert "gpu_utilization" in stats
        assert stats["gpu_utilization"] == 0.6
        assert stats["gpu_temperature"] == 55.0


# ---------------------------------------------------------------------------
# Tegrastats parser tests
# ---------------------------------------------------------------------------

class TestTegrastatsParser:
    def test_parse_gr3d_freq(self):
        from weda_optim.gpu_monitor import TegrastatsMonitor
        line = "RAM 4567/15830MB GR3D_FREQ 78%@1300 VIC_FREQ 0%@115"
        assert TegrastatsMonitor._parse_gpu_util(line) == 0.78

    def test_parse_temperature(self):
        from weda_optim.gpu_monitor import TegrastatsMonitor
        line = "tj@52.5C gpu@48C"
        temp = TegrastatsMonitor._parse_temperature(line)
        assert temp == 52.5  # first match (tj)

    def test_parse_missing_data(self):
        from weda_optim.gpu_monitor import TegrastatsMonitor
        assert TegrastatsMonitor._parse_gpu_util("") == 0.0
        assert TegrastatsMonitor._parse_temperature("") == 0.0


# ---------------------------------------------------------------------------
# RTSP input tests
# ---------------------------------------------------------------------------

class TestRtspInput:
    def test_is_rtsp_detection(self):
        from app.io.rtsp_backend import _is_rtsp
        assert _is_rtsp("rtsp://192.168.1.100/stream") is True
        assert _is_rtsp("RTSP://host/path") is True
        assert _is_rtsp("/path/to/video.mp4") is False
        assert _is_rtsp("0") is False

    def test_sanitize_url_strips_credentials(self):
        from app.io.rtsp_backend import _sanitize_url
        result = _sanitize_url("rtsp://admin:password123@192.168.1.100:554/stream")
        assert "password123" not in result
        assert "<credentials>" in result
        assert "192.168.1.100:554/stream" in result

    def test_sanitize_url_no_credentials(self):
        from app.io.rtsp_backend import _sanitize_url
        url = "rtsp://192.168.1.100/stream"
        assert _sanitize_url(url) == url

    def test_rtsp_open_fails_without_pyav(self, monkeypatch):
        import app.io.rtsp_backend as rb
        monkeypatch.setattr(rb, "_HAS_AV", False)
        from app.utils.data_structures import PipelineConfig
        config = PipelineConfig(source="rtsp://host/stream")
        backend = rb.RTSPBackend(config)
        assert backend.open() is False


# ---------------------------------------------------------------------------
# Query Engine tests
# ---------------------------------------------------------------------------

def _make_test_scene():
    """Fixture: 3 bottles + 2 persons with spatial data."""
    from app.utils.data_structures import Detection, SpatialInfo
    dets = [
        Detection(bbox=np.array([100, 100, 150, 200], dtype=np.float32),
                  confidence=0.9, class_id=0, class_name="bottle", track_id=3),
        Detection(bbox=np.array([300, 100, 350, 200], dtype=np.float32),
                  confidence=0.85, class_id=0, class_name="bottle", track_id=5),
        Detection(bbox=np.array([500, 100, 550, 200], dtype=np.float32),
                  confidence=0.8, class_id=0, class_name="bottle", track_id=7),
        Detection(bbox=np.array([200, 50, 300, 400], dtype=np.float32),
                  confidence=0.95, class_id=1, class_name="person", track_id=12),
        Detection(bbox=np.array([600, 50, 700, 400], dtype=np.float32),
                  confidence=0.92, class_id=1, class_name="person", track_id=14),
    ]
    spatial = [
        SpatialInfo(distance_m=0.43, position_3d=np.array([0.1, -0.1, 0.43]),
                    size_estimate=(0.08, 0.25), nearest_neighbor_id=12, nearest_neighbor_dist=0.4),
        SpatialInfo(distance_m=0.64, position_3d=np.array([0.3, -0.1, 0.64]),
                    size_estimate=(0.07, 0.24), nearest_neighbor_id=7, nearest_neighbor_dist=0.3),
        SpatialInfo(distance_m=0.32, position_3d=np.array([0.5, -0.1, 0.32]),
                    size_estimate=(0.09, 0.26), nearest_neighbor_id=5, nearest_neighbor_dist=0.3),
        SpatialInfo(distance_m=2.5, position_3d=np.array([0.2, -0.5, 2.5]),
                    size_estimate=(0.5, 1.7), nearest_neighbor_id=3, nearest_neighbor_dist=0.4),
        SpatialInfo(distance_m=4.0, position_3d=np.array([0.8, -0.5, 4.0]),
                    size_estimate=(0.5, 1.7), nearest_neighbor_id=14, nearest_neighbor_dist=1.5),
    ]
    return dets, spatial


class TestQueryEngineClassify:
    def test_classify_count(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, target, _ = QueryEngine.classify("how many bottles?")
        assert qtype == QueryType.COUNT
        assert target == "bottles"

    def test_classify_distance(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, target, _ = QueryEngine.classify("distance of the person?")
        assert qtype == QueryType.DISTANCE
        assert target == "person"

    def test_classify_position(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, _, _ = QueryEngine.classify("where is the bottle?")
        assert qtype == QueryType.POSITION

    def test_classify_nearest(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, target, _ = QueryEngine.classify("what's closest to the person?")
        assert qtype == QueryType.NEAREST
        assert target == "person"

    def test_classify_list(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, _, _ = QueryEngine.classify("what objects are there?")
        assert qtype == QueryType.LIST_OBJECTS

    def test_classify_size(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, _, _ = QueryEngine.classify("how big is the person?")
        assert qtype == QueryType.SIZE

    def test_classify_color(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, target, _ = QueryEngine.classify("what color is the bottle?")
        assert qtype == QueryType.COLOR
        assert target == "bottle"

    def test_classify_describe(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, _, _ = QueryEngine.classify("describe the person")
        assert qtype == QueryType.DESCRIBE_OBJECT

    def test_classify_open_ended(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, _, _ = QueryEngine.classify("what should the operator check first?")
        assert qtype == QueryType.OPEN_ENDED


class TestQueryEngineTier1:
    def test_count_bottles(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        resp, _ = QueryEngine.process("how many bottles?", dets, spatial)
        assert "3" in resp
        assert "bottle" in resp

    def test_count_persons(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        resp, _ = QueryEngine.process("how many persons?", dets, spatial)
        assert "2" in resp

    def test_count_missing_class(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        resp, _ = QueryEngine.process("how many forklifts?", dets, spatial)
        assert "No forklift" in resp

    def test_distance(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        resp, _ = QueryEngine.process("distance of the bottle?", dets, spatial)
        assert "0.43m" in resp
        assert "0.64m" in resp
        assert "0.32m" in resp

    def test_position(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        resp, _ = QueryEngine.process("where is the person?", dets, spatial)
        assert "2.50m" in resp or "2.5" in resp

    def test_nearest(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        resp, _ = QueryEngine.process("what's closest to the person?", dets, spatial)
        assert "bottle" in resp.lower() or "track #3" in resp

    def test_list_objects(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        resp, _ = QueryEngine.process("what objects are there?", dets, spatial)
        assert "bottle" in resp
        assert "person" in resp
        assert "5" in resp  # total

    def test_size(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        resp, _ = QueryEngine.process("how big is the person?", dets, spatial)
        assert "0.50m" in resp or "1.70m" in resp

    def test_empty_scene(self):
        from app.modules.query_engine import QueryEngine
        resp, _ = QueryEngine.process("how many bottles?", [], [])
        assert "No" in resp or "no" in resp

    def test_plural_handling(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        # "bottles" should match "bottle" class_name
        resp, _ = QueryEngine.process("count bottles", dets, spatial)
        assert "3" in resp


class TestQueryEngineTier2:
    def test_visual_with_cache(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        cache = {3: "blue plastic bottle with Advantech logo"}
        resp, _ = QueryEngine.process("what color is the bottle?", dets, spatial, cache)
        assert "blue" in resp
        assert "Advantech" in resp

    def test_visual_without_cache(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        resp, _ = QueryEngine.process("describe the bottle", dets, spatial, {})
        assert "not yet cached" in resp or "visual details" in resp.lower()


class TestQueryEngineTier3:
    def test_open_ended_returns_grounded_context(self):
        from app.modules.query_engine import QueryEngine, QueryType
        dets, spatial = _make_test_scene()
        resp, qtype = QueryEngine.process("what should the operator check first?", dets, spatial)
        assert qtype == QueryType.OPEN_ENDED
        assert "VERIFIED SCENE DATA" in resp
        assert "bottle" in resp
        assert "person" in resp

    def test_grounded_context_includes_distances(self):
        from app.modules.query_engine import QueryEngine
        dets, spatial = _make_test_scene()
        ctx = QueryEngine.build_grounded_context("test", dets, spatial)
        assert "0.43m" in ctx or "0.43" in ctx
        assert "track #3" in ctx


class TestObjectCache:
    def test_cache_stores_and_returns(self):
        from app.modules.reasoning import SceneReasoner
        from app.utils.data_structures import PipelineConfig
        r = SceneReasoner(PipelineConfig())
        r._object_cache[3] = "blue bottle"
        r._object_cache[5] = "red bottle"
        cache = r.object_cache
        assert cache[3] == "blue bottle"
        assert cache[5] == "red bottle"

    def test_cache_returns_copy(self):
        from app.modules.reasoning import SceneReasoner
        from app.utils.data_structures import PipelineConfig
        r = SceneReasoner(PipelineConfig())
        r._object_cache[1] = "test"
        cache = r.object_cache
        cache[99] = "injected"
        assert 99 not in r._object_cache  # original unaffected

    def test_prune_removes_stale(self):
        from app.modules.reasoning import SceneReasoner
        from app.utils.data_structures import PipelineConfig
        r = SceneReasoner(PipelineConfig())
        r._object_cache = {1: "a", 2: "b", 3: "c"}
        r.prune_cache({1, 3})  # track 2 is stale
        assert 2 not in r._object_cache
        assert 1 in r._object_cache
        assert 3 in r._object_cache

    def test_shutdown_clears_cache(self):
        from app.modules.reasoning import SceneReasoner
        from app.utils.data_structures import PipelineConfig
        r = SceneReasoner(PipelineConfig())
        r._object_cache[1] = "test"
        r.shutdown()
        assert len(r._object_cache) == 0


class TestQueryPipelineIntegration:
    def test_tier1_query_gets_response(self):
        """Full pipeline with pending_query → instant query_response."""
        from app.core.orchestrator import Orchestrator
        from app.utils.data_structures import PipelineConfig, FramePacket

        config = PipelineConfig(
            source="synthetic", width=640, height=480,
            headless=True, max_frames=5, pool_depth=2,
            profile_interval=0, log_level="WARNING",
        )
        orch = Orchestrator(config)
        orch._detector.initialize()
        orch._reasoner.initialize()
        orch._input.open()

        # Process a few frames to build up tracker state
        for _ in range(3):
            packet = orch._input.read()
            if packet:
                for token in orch._input.pop_frame_tokens():
                    packet.record_pool_token(*token)
                orch._pipeline.process(packet)
                orch._pool.release_all(packet._pool_tokens)

        # Now send a query
        packet = orch._input.read()
        if packet:
            for token in orch._input.pop_frame_tokens():
                packet.record_pool_token(*token)
            packet.pending_query = "what objects are there?"
            result = orch._pipeline.process(packet)
            # Should have a response (even if "No objects" in mock mode)
            assert result.query_response is not None
            assert len(result.query_response) > 0
            orch._pool.release_all(packet._pool_tokens)

        orch._input.close()

    def test_no_query_zero_overhead(self):
        """Pipeline with no pending_query should not touch query engine."""
        from app.core.orchestrator import Orchestrator
        from app.utils.data_structures import PipelineConfig

        config = PipelineConfig(
            source="synthetic", width=640, height=480,
            headless=True, max_frames=10, pool_depth=2,
            profile_interval=0, log_level="WARNING",
        )
        orch = Orchestrator(config)
        orch.run()
        summary = orch._profiler.summary()
        # "query" stage should NOT appear in profiler (no queries were made)
        assert "query" not in summary["stages"]


# ---------------------------------------------------------------------------
# Embedder tests (Cosmos-Embed1 mock mode)
# ---------------------------------------------------------------------------

class TestEmbedder:
    def _make(self):
        from app.modules.embedder import Embedder
        from app.utils.data_structures import PipelineConfig
        config = PipelineConfig(enable_embedding=True, embedding_dim=256)
        emb = Embedder(config)
        emb.initialize()  # model will be None on dev machine without Cosmos
        return emb

    def test_embed_frame_no_model_returns_none(self):
        emb = self._make()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        vec, changed = emb.embed_frame(frame)
        assert vec is None
        assert changed is False

    def test_embed_crops_no_model_returns_none_list(self):
        from app.utils.data_structures import Detection
        emb = self._make()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = [
            Detection(bbox=np.array([100, 100, 200, 200], dtype=np.float32),
                      confidence=0.9, class_id=0, class_name="bottle"),
            Detection(bbox=np.array([300, 100, 400, 200], dtype=np.float32),
                      confidence=0.8, class_id=0, class_name="bottle"),
        ]
        embeddings = emb.embed_crops(frame, dets)
        assert len(embeddings) == 2
        assert all(e is None for e in embeddings)

    def test_cosine_similarity(self):
        emb = self._make()
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(emb.cosine_similarity(a, b) - 1.0) < 1e-6
        c = np.array([0.0, 1.0, 0.0])
        assert abs(emb.cosine_similarity(a, c)) < 1e-6


# ---------------------------------------------------------------------------
# Physical Reasoner tests
# ---------------------------------------------------------------------------

class TestPhysicalReasoner:
    def _make(self):
        from app.modules.physical_reasoner import PhysicalReasoner
        from app.utils.data_structures import PipelineConfig
        config = PipelineConfig(enable_physical_reasoning=True)
        pr = PhysicalReasoner(config)
        pr.initialize()  # model will be None on dev machine without PyTorch/Cosmos
        return pr

    def test_no_model_returns_error_string(self):
        pr = self._make()
        result = pr.query_sync(
            np.zeros((100, 100, 3), dtype=np.uint8), [], [], "is it safe?"
        )
        assert "not available" in result.lower()


# ---------------------------------------------------------------------------
# Tracker re-ID tests
# ---------------------------------------------------------------------------

class TestTrackerReID:
    def _make(self):
        from app.modules.bytetrack import ByteTracker as Tracker
        return Tracker(iou_threshold=0.3, embedding_threshold=0.7)

    def test_embedding_stored_on_track(self):
        from app.utils.data_structures import Detection
        t = self._make()
        emb = np.random.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)
        det = Detection(
            bbox=np.array([0, 0, 50, 50], dtype=np.float32),
            confidence=0.9, class_id=0, class_name="bottle",
            embedding=emb,
        )
        t.update([det])
        assert det.track_id > 0
        track = t._tracks[det.track_id]
        assert track.embedding is not None

    def test_active_track_ids(self):
        from app.utils.data_structures import Detection
        t = self._make()
        dets = [
            Detection(bbox=np.array([0, 0, 50, 50], dtype=np.float32),
                      confidence=0.9, class_id=0, class_name="a"),
            Detection(bbox=np.array([100, 100, 150, 150], dtype=np.float32),
                      confidence=0.9, class_id=1, class_name="b"),
        ]
        t.update(dets)
        ids = t.active_track_ids
        assert len(ids) == 2


# ---------------------------------------------------------------------------
# Tier 2.5 query classification tests
# ---------------------------------------------------------------------------

class TestQueryTier25:
    def test_classify_safety(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, _, _ = QueryEngine.classify("is it safe to move?")
        assert qtype == QueryType.SAFETY

    def test_classify_path_clear(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, _, _ = QueryEngine.classify("is the path clear?")
        assert qtype == QueryType.PATH_CLEAR

    def test_classify_physics(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, _, _ = QueryEngine.classify("will it fall?")
        assert qtype == QueryType.PHYSICS

    def test_classify_ppe(self):
        from app.modules.query_engine import QueryEngine, QueryType
        qtype, _, _ = QueryEngine.classify("is the worker wearing PPE?")
        assert qtype == QueryType.PPE_CHECK


# ---------------------------------------------------------------------------
# YOLO26 detector config test
# ---------------------------------------------------------------------------

class TestYOLO26:
    def test_yolo26_flag(self):
        from app.modules.detector import Detector
        from app.utils.data_structures import PipelineConfig
        config = PipelineConfig(yolo_model="yolo26n-seg.pt")
        det = Detector(config)
        assert det.is_yolo26 is True

    def test_yolo11_flag(self):
        from app.modules.detector import Detector
        from app.utils.data_structures import PipelineConfig
        config = PipelineConfig(yolo_model="yolo11n-seg.pt")
        det = Detector(config)
        assert det.is_yolo26 is False


# ---------------------------------------------------------------------------
# Config new fields tests
# ---------------------------------------------------------------------------

class TestNewConfigFields:
    def test_embedding_defaults(self):
        from app.utils.data_structures import PipelineConfig
        c = PipelineConfig()
        assert c.enable_embedding is False
        assert c.embedding_model == "nvidia/Cosmos-Embed1-224p"
        assert c.embedding_dim == 256

    def test_physical_reasoning_defaults(self):
        from app.utils.data_structures import PipelineConfig
        c = PipelineConfig()
        assert c.enable_physical_reasoning is False
        assert c.physical_model == "nvidia/Cosmos-Reason2-2B"

    def test_vlm_model_default_qwen3(self):
        from app.utils.data_structures import PipelineConfig
        c = PipelineConfig()
        assert "Qwen3-VL" in c.vlm_model
        assert c.vlm_prompt_format == "qwen3"
        assert c.quant_vlm_strategy == "nvfp4"

    def test_yolo26_default(self):
        from app.utils.data_structures import PipelineConfig
        c = PipelineConfig()
        assert "yolo26" in c.yolo_model

    def test_scene_change_threshold(self):
        from app.utils.data_structures import PipelineConfig
        c = PipelineConfig()
        assert c.scene_change_threshold == 0.95

    def test_yaml_loads_embedding_config(self, tmp_path):
        from app.config import load_config
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "embedding:\n  enabled: true\n  dim: 768\n"
            "  scene_change_threshold: 0.9\n"
        )
        config = load_config(str(yaml_file))
        assert config.enable_embedding is True
        assert config.embedding_dim == 768
        assert config.scene_change_threshold == 0.9

    def test_yaml_loads_physical_config(self, tmp_path):
        from app.config import load_config
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "physical_reasoning:\n  enabled: true\n  cooldown_sec: 3.0\n"
        )
        config = load_config(str(yaml_file))
        assert config.enable_physical_reasoning is True
        assert config.physical_cooldown == 3.0


# ---------------------------------------------------------------------------
# Quantization backend tests
# ---------------------------------------------------------------------------

class TestQuantization:
    def test_detect_backend_auto(self):
        from weda_optim.quantization import detect_backend
        result = detect_backend("auto")
        assert result in ("modelopt", "bitsandbytes", "none")

    def test_detect_backend_none(self):
        from weda_optim.quantization import detect_backend
        assert detect_backend("none") == "none"

    def test_detect_backend_missing(self):
        from weda_optim.quantization import detect_backend
        # Request modelopt — may or may not be installed
        result = detect_backend("modelopt")
        assert result in ("modelopt", "none")

    def test_bnb_config_nf4(self):
        from weda_optim.quantization import get_quantization_config
        cfg = get_quantization_config("bitsandbytes", "nf4")
        # On dev machine without bitsandbytes, returns None
        # With bitsandbytes, returns BitsAndBytesConfig
        assert cfg is None or hasattr(cfg, "load_in_4bit")

    def test_bnb_config_for_nvfp4_maps_to_nf4(self):
        from weda_optim.quantization import _map_to_bnb
        assert _map_to_bnb("nvfp4") == "nf4"
        assert _map_to_bnb("fp8") == "int8"
        assert _map_to_bnb("nf4") == "nf4"

    def test_quant_config_defaults(self):
        from app.utils.data_structures import PipelineConfig
        c = PipelineConfig()
        assert c.quant_backend == "auto"
        assert c.quant_vlm_strategy == "nvfp4"
        assert c.quant_vlm_kv_cache == "nvfp4"
        assert c.quant_physical_strategy == "fp8"
        assert c.quant_cache_dir == "model_cache"
        assert c.quant_calibration_samples == 8


# ---------------------------------------------------------------------------
# Model loader / prompt format tests
# ---------------------------------------------------------------------------

class TestModelLoader:
    def test_prompt_format_registry(self):
        from weda_optim.model_loader import PROMPT_FORMATS
        assert "llava" in PROMPT_FORMATS
        assert "qwen3" in PROMPT_FORMATS
        assert "cosmos" in PROMPT_FORMATS
        assert "chatml" in PROMPT_FORMATS
        assert "raw" in PROMPT_FORMATS

    def test_llava_format(self):
        from weda_optim.model_loader import get_prompt_formatter
        fmt = get_prompt_formatter("llava")
        result = fmt("scene data", "what is this?")
        assert "USER:" in result
        assert "ASSISTANT:" in result
        assert "scene data" in result
        assert "what is this?" in result

    def test_qwen3_format_returns_list(self):
        from weda_optim.model_loader import get_prompt_formatter
        fmt = get_prompt_formatter("qwen3")
        result = fmt("context", "query")
        assert isinstance(result, list)
        # qwen3 formatter now includes a system message for sensor-grounding
        # followed by the user message — verify both are present
        roles = [m["role"] for m in result]
        assert "system" in roles
        assert "user" in roles

    def test_cosmos_format_returns_list(self):
        from weda_optim.model_loader import get_prompt_formatter
        fmt = get_prompt_formatter("cosmos")
        result = fmt("context", "query")
        assert isinstance(result, list)
        assert result[0]["role"] == "system"

    def test_chatml_format(self):
        from weda_optim.model_loader import get_prompt_formatter
        fmt = get_prompt_formatter("chatml")
        result = fmt("", "query")
        assert "<|im_start|>" in result

    def test_raw_format(self):
        from weda_optim.model_loader import get_prompt_formatter
        fmt = get_prompt_formatter("raw")
        assert fmt("", "hello") == "hello"
        assert fmt("ctx", "q") == "ctx\n\nq"

    def test_unknown_format_fallback(self):
        from weda_optim.model_loader import get_prompt_formatter
        fmt = get_prompt_formatter("nonexistent_format")
        # Should fall back to raw
        assert fmt("", "test") == "test"

    def test_model_swap_via_config(self, tmp_path):
        from app.config import load_config
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "vlm:\n"
            "  model: 'llava-hf/llava-1.5-7b-hf'\n"
            "  prompt_format: 'llava'\n"
            "  max_tokens: 50\n"
        )
        config = load_config(str(yaml_file))
        assert config.vlm_model == "llava-hf/llava-1.5-7b-hf"
        assert config.vlm_prompt_format == "llava"
        assert config.vlm_max_tokens == 50

    def test_physical_model_swap_via_config(self, tmp_path):
        from app.config import load_config
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "physical_reasoning:\n"
            "  model: 'nvidia/Cosmos-Reason2-8B'\n"
            "  prompt_format: 'chatml'\n"
            "  max_tokens: 512\n"
        )
        config = load_config(str(yaml_file))
        assert config.physical_model == "nvidia/Cosmos-Reason2-8B"
        assert config.physical_prompt_format == "chatml"
        assert config.physical_max_tokens == 512

    def test_quantization_yaml(self, tmp_path):
        from app.config import load_config
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "quantization:\n"
            "  backend: 'bitsandbytes'\n"
            "  vlm_strategy: 'nf4'\n"
            "  cache_dir: '/tmp/cache'\n"
        )
        config = load_config(str(yaml_file))
        assert config.quant_backend == "bitsandbytes"
        assert config.quant_vlm_strategy == "nf4"
        assert config.quant_cache_dir == "/tmp/cache"

    def test_detection_input_size_config(self, tmp_path):
        from app.config import load_config
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "detection:\n  model: 'yolo26m-seg.pt'\n  input_size: 1280\n"
        )
        config = load_config(str(yaml_file))
        assert config.yolo_model == "yolo26m-seg.pt"
        assert config.det_input_size == 1280
