import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import socket
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_message import OscMessage
import main as receiver


SOURCE = ("127.0.0.1", 20001)
OTHER_SOURCE = ("127.0.0.1", 20002)


def bone(index, parent):
    return [index, parent, f"Bone_{index}", 1., 2., 3., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.]


def motion_args(frame, bones, chunks, chunk, total):
    return ["2026-10-01 00:00:00.000", "MOVINMan", frame, chunks, chunk, total, len(bones), *sum(bones, [])]


def packet(address, args):
    message = OscMessageBuilder(address=address)
    for value in args:
        message.add_arg(value)
    return message.build().dgram


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.
        self.clock = patch.object(receiver.time, "monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.state = receiver.SharedState(timeout=1.)

    def motion(self, args, source):
        receiver.parse_motion("/MOVIN/Frame", *args, state=self.state, source=source)

    def cloud(self, args, source):
        receiver.parse_point_cloud("/MOVIN/PointCloud", *args, state=self.state, source=source)

    def test_reversed_motion_chunks_and_duplicate_do_not_reapply(self):
        second = motion_args(10, [bone(1, 0)], 2, 1, 2)
        self.motion(second, SOURCE)
        self.motion(second, SOURCE)
        self.assertFalse(self.state.latest_skeletons)
        self.motion(motion_args(10, [bone(0, -1)], 2, 0, 2), SOURCE)
        original = self.state.latest_skeletons["MOVINMan"]
        self.now += .1
        self.motion(second, SOURCE)
        self.assertIs(self.state.latest_skeletons["MOVINMan"], original)
        self.assertEqual(self.state.last_update["MOVINMan"], 1000.)
        np.testing.assert_allclose(original[1].world_position, [-2, 4, 6])

    def test_older_frames_never_replace_healthy_motion(self):
        self.motion(motion_args(300, [bone(0, -1)], 1, 0, 1), SOURCE)
        for frame in (299, 100, 0, 300):
            self.now += .1
            self.motion(motion_args(frame, [bone(0, -1)], 1, 0, 1), SOURCE)
            self.assertEqual(self.state.motion_streams["MOVINMan"].latest, 300)

    def test_restart_waits_one_second_without_progress(self):
        self.motion(motion_args(100, [bone(0, -1)], 1, 0, 1), SOURCE)
        self.now += .9
        self.motion(motion_args(0, [bone(0, -1)], 1, 0, 1), SOURCE)
        self.assertEqual(self.state.motion_streams["MOVINMan"].latest, 100)
        self.now += .11
        self.motion(motion_args(1, [bone(0, -1)], 1, 0, 1), SOURCE)
        self.assertEqual(self.state.motion_streams["MOVINMan"].latest, 1)

    def test_incomplete_frames_are_bounded_and_expire_without_new_packets(self):
        for frame in range(600):
            self.motion(motion_args(frame, [bone(0, -1)], 2, 0, 2), SOURCE)
            self.now += 1 / 60
            self.assertLessEqual(len(self.state.motion_streams["MOVINMan"].frames), receiver.MAX_PENDING_FRAMES)
        self.now += .6
        self.state.expire(self.now)
        self.assertFalse(self.state.motion_streams["MOVINMan"].frames)

    def test_sparse_indices_keep_all_transmitted_bones(self):
        records = [bone(i, -1 if i == 0 else 0) for i in range(50) if i not in (2, 3)]
        self.motion(motion_args(1, records, 1, 0, 48), SOURCE)
        actual = self.state.latest_skeletons["MOVINMan"]
        self.assertEqual([b.bone_index for b in actual], [b[0] for b in records])

    def test_one_bone_per_chunk_is_allowed_for_large_rigs(self):
        count = receiver.MAX_BONES
        self.motion(motion_args(1, [bone(count-1, 0)], count, count-1, count), SOURCE)
        self.assertEqual(self.state.motion_streams['MOVINMan'].frames[1].received, 1)

    def test_bad_header_and_payload_types_are_rejected_before_state_changes(self):
        original = motion_args(1, [bone(0, -1)], 1, 0, 1)
        for index, value in [(2,-1), (2,1.), (3,0), (3,3000), (4,99), (5,10000000), (5,-1),
                             (6,-1), (6,2), (1,""), (1,"x"*257), (7,-1), (8,0), (10,float('nan')),
                             (10,float('inf')), (10,"1"), (10,1)]:
            with self.subTest(index=index, value=value):
                args = original.copy()
                args[index] = value
                with self.assertRaises(ValueError):
                    self.motion(args, SOURCE)
                self.assertFalse(self.state.latest_skeletons)
        for args in (original[:6], original[:-1], original + [1.]):
            with self.assertRaises(ValueError):
                self.motion(args, SOURCE)

    def test_zero_rotation_and_hierarchy_errors_are_rejected(self):
        zero_rotation = bone(0, -1)
        zero_rotation[10:14] = [0.] * 4
        cases = [[zero_rotation], [bone(0,1), bone(1,0)], [bone(0,9)],
                 [bone(0,-1),bone(0,-1)], [bone(0,-1),[1,0,"Bone_0",*bone(1,0)[3:]]]]
        for records in cases:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    self.motion(motion_args(1, records, 1, 0, len(records)), SOURCE)
                self.assertFalse(self.state.latest_skeletons)

    def test_deep_hierarchy_does_not_use_python_recursion(self):
        records = [bone(i, i - 1) for i in range(1100)]
        self.motion(motion_args(1, records, 1, 0, len(records)), SOURCE)
        self.assertEqual(len(self.state.latest_skeletons["MOVINMan"]), 1100)

    def test_large_finite_quaternion_is_normalized_without_overflow(self):
        record = bone(0, -1)
        record[10:14] = [1e30, 0., 0., 1e30]
        self.motion(motion_args(1, [record], 1, 0, 1), SOURCE)
        q = self.state.latest_skeletons["MOVINMan"][0].world_rotation
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1., places=5)

    def test_header_disagreement_discards_partial_frame(self):
        self.motion(motion_args(1, [bone(0,-1)], 2, 0, 2), SOURCE)
        args = motion_args(1, [bone(1,0)], 2, 1, 3)
        with self.assertRaises(ValueError):
            self.motion(args, SOURCE)
        self.assertFalse(self.state.motion_streams["MOVINMan"].frames)

    def test_complete_but_missing_bones_is_rejected(self):
        with self.assertRaises(ValueError):
            self.motion(motion_args(1, [bone(0,-1)], 1, 0, 2), SOURCE)
        self.assertFalse(self.state.latest_skeletons)

    def test_other_endpoint_cannot_mix_chunks_and_can_take_over_after_silence(self):
        self.motion(motion_args(1, [bone(0,-1)], 2, 0, 2), SOURCE)
        self.motion(motion_args(1, [bone(1,0)], 2, 1, 2), OTHER_SOURCE)
        self.assertFalse(self.state.latest_skeletons)
        self.motion(motion_args(1, [bone(1,0)], 2, 1, 2), SOURCE)
        self.assertEqual(len(self.state.latest_skeletons["MOVINMan"]), 2)
        self.now += 1.01
        self.motion(motion_args(0, [bone(0,-1)], 1, 0, 1), OTHER_SOURCE)
        self.assertEqual(self.state.source, OTHER_SOURCE)
        self.assertEqual(len(self.state.latest_skeletons["MOVINMan"]), 1)

    def test_actor_count_is_bounded(self):
        for i in range(receiver.MAX_ACTORS):
            args = motion_args(1, [bone(0,-1)], 2, 0, 2)
            args[1] = str(i)
            self.motion(args, SOURCE)
        args[1] = "extra"
        with self.assertRaises(ValueError):
            self.motion(args, SOURCE)

    def test_pointcloud_reordering_duplicates_and_expiry(self):
        second = [10,2,1,2,1,4.,5.,6.]
        self.cloud(second, SOURCE)
        self.cloud(second, SOURCE)
        self.cloud([10,2,0,2,1,1.,2.,3.], SOURCE)
        np.testing.assert_allclose(self.state.latest_points, [[-1,2,3],[-4,5,6]])
        self.assertFalse(self.state.latest_points.flags.writeable)
        original = self.state.latest_points
        self.cloud([9,1,0,1,1,7.,8.,9.], SOURCE)
        self.assertIs(self.state.latest_points, original)
        self.assertIs(self.state.snapshot()[1], original)
        self.now += 1.01
        self.assertEqual(len(self.state.snapshot()[1]), 0)
        self.cloud([0,1,0,1,1,7.,8.,9.], SOURCE)
        np.testing.assert_allclose(self.state.latest_points, [[-7,8,9]])

    def test_empty_pointcloud_clears_previous_frame(self):
        self.cloud([1,1,0,1,1,1.,2.,3.], SOURCE)
        self.cloud([2,0,0,1,0], SOURCE)
        self.assertEqual(len(self.state.latest_points), 0)

    def test_invalid_pointcloud_packets(self):
        for args in ([1,1,99,1,1,1.,2.,3.], [1,10000001,0,1,1,1.,2.,3.],
                     [1,1,0,1,1,float('nan'),2.,3.], [1,1,0,1,1,1,2,3],
                     [1,1,0,1,-1], [1,2,0,1,1,1.,2.,3.]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.cloud(args, SOURCE)
        self.assertEqual(len(self.state.latest_points), 0)

    def test_incomplete_clouds_are_bounded_and_expire(self):
        for frame in range(100):
            self.cloud([frame,2,0,2,1,1.,2.,3.], SOURCE)
        self.assertLessEqual(len(self.state.point_stream.frames), receiver.MAX_PENDING_FRAMES)
        self.now += .6
        self.state.expire(self.now)
        self.assertFalse(self.state.point_stream.frames)

    def test_dispatcher_logs_invalid_packet_and_recovers(self):
        dispatcher = receiver.create_dispatcher(self.state)
        with self.assertLogs(receiver.logger, level="WARNING") as logs:
            for _ in range(10):
                dispatcher.call_handlers_for_packet(packet('/MOVIN/Frame', ['invalid']), SOURCE)
        self.assertEqual(len(logs.output), 1)
        args = motion_args(1, [bone(0,-1)], 1, 0, 1)
        dispatcher.call_handlers_for_packet(packet('/MOVIN/Frame', args), SOURCE)
        self.assertIn('MOVINMan', self.state.latest_skeletons)

    def test_fps_counts_only_completed_frames_and_is_independent_of_viewer(self):
        for frame in range(61):
            self.now = 1000. + frame / 60
            first = motion_args(frame, [bone(0,-1)], 2, 0, 2)
            self.motion(first, SOURCE)
            self.motion(first, SOURCE)
            self.motion(motion_args(frame, [bone(1,0)], 2, 1, 2), SOURCE)
            self.cloud([frame,1,0,1,1,1.,2.,3.], SOURCE)
            if frame % 2 == 0:
                self.state.viewer_rate.add(self.now)
        self.assertAlmostEqual(self.state.motion_rates['MOVINMan'].read(self.now)[0], 60.)
        self.assertAlmostEqual(self.state.cloud_rate.read(self.now)[0], 60.)
        self.assertAlmostEqual(self.state.viewer_rate.read(self.now)[0], 30.)
        self.now += 1.01
        self.assertEqual(self.state.cloud_rate.read(self.now)[0], 0.)
        self.assertEqual(self.state.viewer_rate.read(self.now)[0], 0.)
        self.motion(motion_args(0, [bone(0,-1)], 1, 0, 1), OTHER_SOURCE)
        self.assertEqual(self.state.motion_rates['MOVINMan'].read(self.now), (0., 0.))
        self.assertEqual(self.state.cloud_rate.read(self.now), (0., -1.))

    def test_motion_fps_is_per_actor_and_ages_out(self):
        for frame in range(61):
            self.now = 1000. + frame / 60
            self.motion(motion_args(frame, [bone(0,-1)], 1, 0, 1), SOURCE)
            if frame % 2 == 0:
                args = motion_args(frame, [bone(0,-1)], 1, 0, 1)
                args[1] = 'Other Actor'
                self.motion(args, SOURCE)
        self.assertAlmostEqual(self.state.motion_rates['MOVINMan'].read(self.now)[0], 60.)
        self.assertAlmostEqual(self.state.motion_rates['Other Actor'].read(self.now)[0], 30.)
        self.now += 1.01
        self.state.snapshot()
        self.assertEqual(self.state.motion_rates, {})

    def test_frame_rate_storage_is_bounded_and_never_reports_stale_fps(self):
        rate = receiver.FrameRate()
        self.assertEqual(rate.read(self.now), (0., -1.))
        for i in range(5000):
            rate.add(self.now + i / 10000)
        self.assertLessEqual(len(rate.times), 4096)
        self.assertEqual(rate.read(self.now + 2)[0], 0.)


class UDPTests(unittest.TestCase):
    def test_status_reply_uses_shared_socket_and_does_not_claim_other_senders_data(self):
        state = receiver.SharedState(timeout=1)
        server = receiver.ReceiverServer(('127.0.0.1', 0), state)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        thread.start()
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        response = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        response.bind(('127.0.0.1', 0))
        response.settimeout(2)
        port = response.getsockname()[1]
        token = '0123456789abcdef' * 2
        try:
            udp.sendto(packet('/MOVIN/Frame', motion_args(1, [bone(0,-1)], 1, 0, 1)), server.server_address)
            udp.sendto(packet('/MOVIN/PointCloud', [1,1,0,1,1,1.,2.,3.]), server.server_address)
            udp.sendto(packet('/MOVIN/OSC/Status/Request', [token, port, 'MOVINMan']), server.server_address)
            data, address = response.recvfrom(2048)
            reply = OscMessage(data)
            self.assertEqual(address, server.server_address)
            self.assertEqual(reply.address, '/MOVIN/OSC/Status')
            self.assertEqual(reply.params[:3], [token, 1, 'MOVINMan'])
            self.assertEqual(reply.params[9], 1)
            self.assertGreaterEqual(reply.params[6], 0.)
            self.assertGreaterEqual(reply.params[7], 0.)
            self.assertEqual(reply.params[8], -1.)
            heard = state.source_heard_at
            response.sendto(packet('/MOVIN/OSC/Status/Request', [token, port, 'MOVINMan']), server.server_address)
            other = OscMessage(response.recvfrom(2048)[0]).params
            self.assertEqual(other[3:5], [0., 0.])
            self.assertEqual(other[6:8], [-1., -1.])
            self.assertEqual(other[9], 0)
            self.assertEqual(state.source_heard_at, heard)
            with self.assertLogs(receiver.logger, level='WARNING'):
                udp.sendto(packet('/MOVIN/OSC/Status/Request', ['bad', port, 'MOVINMan']), server.server_address)
                udp.sendto(packet('/MOVIN/OSC/Status/Request', [token, port, 'MOVINMan']), server.server_address)
                recovered = OscMessage(response.recvfrom(2048)[0]).params
                self.assertEqual(recovered[0], token)
        finally:
            udp.close()
            response.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_real_udp_accepts_legacy_large_packet_then_sparse_frame_and_cloud(self):
        state = receiver.SharedState(timeout=1)
        server = receiver.ReceiverServer(('127.0.0.1', 0), state)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        thread.start()
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            records = [bone(i, -1 if i == 0 else 0) for i in range(22)]
            for record in records:
                record[2] += '骨' * 200
            data = packet('/MOVIN/Frame', motion_args(1, records, 1, 0, 22))
            self.assertGreater(len(data), 8192)
            udp.sendto(data, server.server_address)
            until = time.monotonic() + 3
            while 'MOVINMan' not in state.snapshot()[0] and time.monotonic() < until:
                time.sleep(.01)
            self.assertEqual(len(state.snapshot()[0]['MOVINMan']), 22)
            for args in (motion_args(2, [bone(49,0)], 2, 1, 2), motion_args(2, [bone(0,-1)], 2, 0, 2)):
                udp.sendto(packet('/MOVIN/Frame', args), server.server_address)
            udp.sendto(packet('/MOVIN/PointCloud', [1,1,0,1,1,1.,2.,3.]), server.server_address)
            until = time.monotonic() + 3
            while time.monotonic() < until:
                bones, points = state.snapshot()
                if len(bones['MOVINMan']) == 2 and len(points) == 1:
                    break
                time.sleep(.01)
            self.assertEqual([b.bone_index for b in bones['MOVINMan']], [0,49])
            np.testing.assert_allclose(points, [[-1,2,3]])
        finally:
            udp.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
