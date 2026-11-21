"""inkling2_vision_copy.py offline, on a small fake shard 24: the made file, one maker a folder, publishing only with
the pin, and an install that never replaces a file that appeared while it ran. python3 -I inkling2_vision_copy_test.py"""
import argparse
import fcntl
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling2_vision_copy as vc  # noqa: E402


def fake_checkpoint(folder: str) -> bytes:
    """A shard 24 with a text tensor, three tower tensors in one span, and another text tensor; the tower's bytes."""
    names = [("model.llm.a", b"A" * 10), ("model.visual.x", b"x" * 6), ("model.visual.y", b"y" * 4), ("model.visual.z", b"z" * 2), ("model.llm.b", b"B" * 3)]
    meta, at, data = {"__metadata__": {"format": "pt"}}, 0, b""
    for n, b in names:
        meta[n] = {"dtype": "U8", "shape": [len(b)], "data_offsets": [at, at + len(b)]}
        at += len(b)
        data += b
    head = json.dumps(meta).encode()
    with open(os.path.join(folder, vc.SHARD), "wb") as f:
        f.write(struct.pack("<Q", len(head)) + head + data)
    return b"x" * 6 + b"y" * 4 + b"z" * 2


class VisionCopy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "ckpt")
        self.dest = os.path.join(self.tmp.name, "rank0")
        os.makedirs(self.src)
        os.makedirs(self.dest)
        self.tower = fake_checkpoint(self.src)
        self.patch = mock.patch.multiple(vc, TENSORS=3, BYTES=12)
        self.patch.start()
        self.final = os.path.join(self.dest, vc.OUT)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def args(self, sha=None):
        return argparse.Namespace(http=None, src=self.src, dest=self.dest, sha256=sha)

    def pin(self) -> str:
        # build once without a pin: the staging file's sha256, nothing published
        self.assertEqual(vc.make(self.args()), 5)
        self.assertFalse(os.path.exists(self.final))
        return vc.sha256_file(self.final + ".part")

    def test_made_file_holds_the_tower_alone_with_rebased_offsets(self):
        pin = self.pin()
        self.assertEqual(vc.make(self.args(pin)), 0)
        self.assertFalse(os.path.exists(self.final + ".part"))
        with open(self.final, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            head = json.loads(f.read(n))
            self.assertEqual(n % 8, 0)
            self.assertEqual(f.read(), self.tower)
        self.assertEqual(sorted(k for k in head if k != "__metadata__"), ["model.visual.x", "model.visual.y", "model.visual.z"])
        self.assertEqual(head["model.visual.y"]["data_offsets"], [6, 10])
        self.assertEqual(vc.make(self.args(pin)), 0)  # again: already there
        self.assertEqual(vc.check(argparse.Namespace(dest=self.dest, sha256=pin)), 0)

    def test_without_a_pin_nothing_is_published(self):
        self.assertEqual(vc.make(self.args()), 5)
        self.assertFalse(os.path.exists(self.final))

    def test_a_wrong_pin_publishes_nothing(self):
        self.assertEqual(vc.make(self.args("0" * 64)), 4)
        self.assertFalse(os.path.exists(self.final))

    def test_a_second_maker_stops_while_the_first_holds_the_folder(self):
        pin = self.pin()
        part = self.final + ".part"
        before = open(part, "rb").read()
        held = os.open(self.final + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(vc.make(self.args(pin)), 6)
            self.assertEqual(open(part, "rb").read(), before)  # the staging file untouched
            self.assertFalse(os.path.exists(self.final))
        finally:
            os.close(held)
        self.assertEqual(vc.make(self.args(pin)), 0)

    def test_a_file_that_appears_during_the_make_is_never_replaced(self):
        pin = self.pin()
        real = vc.install

        def racing(part, final):
            with open(final, "wb") as f:  # another writer's file lands between the check and the install
                f.write(b"theirs")
            return real(part, final)

        with mock.patch.object(vc, "install", racing):
            self.assertEqual(vc.make(self.args(pin)), 3)
        self.assertEqual(open(self.final, "rb").read(), b"theirs")
        self.assertTrue(os.path.exists(self.final + ".part"))
        self.assertFalse(vc.install(self.final + ".part", self.final))
        self.assertEqual(hashlib.sha256(open(self.final + ".part", "rb").read()).hexdigest(), pin)

    def test_a_publish_cut_between_link_and_unlink_leaves_no_writable_alias(self):
        pin = self.pin()
        self.assertEqual(vc.make(self.args(pin)), 0)
        part = self.final + ".part"
        os.link(self.final, part)  # the link happened, the unlink did not
        self.assertEqual(vc.make(self.args(pin)), 0)  # already published; the alias is gone
        self.assertFalse(os.path.exists(part))
        self.assertEqual(os.stat(self.final).st_nlink, 1)
        # the published file moved away, its alias left as .part: a new make must not write through the alias
        os.link(self.final, part)
        kept = os.path.join(self.dest, "kept.safetensors")
        os.rename(self.final, kept)
        before = open(kept, "rb").read()
        with open(kept, "r+b") as f:  # make the staging look like a resumable, wrong-content .part
            f.seek(len(before) - 1)
            f.write(b"!")
        self.assertEqual(vc.make(self.args(pin)), 0)
        self.assertEqual(open(kept, "rb").read()[-1:], b"!")  # the other name's bytes untouched
        self.assertEqual(os.stat(kept).st_nlink, 1)
        self.assertEqual(vc.sha256_file(self.final), pin)


if __name__ == "__main__":
    unittest.main()
