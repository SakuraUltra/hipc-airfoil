"""Independent fidelity checks for recorded-frame rendering, without image libraries."""

import base64
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_vtk import Field, Grid, compare_grids
from demo_geometry import fluid_mask
from demo_view import write_demo_page

SVG = "{http://www.w3.org/2000/svg}"
MASK_RGB = (226, 231, 233)
# Independently specified scale landmarks: negative end, midpoint, quarter,
# zero, positive quarter, midpoint, and positive end.
PALETTE = {
    -1: (26, 77, 115), -.5: (135, 162, 180), -.25: (190, 204, 213),
    0: (244, 247, 246), .25: (236, 211, 203), .5: (228, 175, 160),
    1: (213, 103, 73),
}


class TimelineReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.parts = []

    def handle_starttag(self, tag, attributes):
        if tag == "script" and dict(attributes).get("id") == "timeline-data":
            self.recording = True

    def handle_data(self, data):
        if self.recording:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.recording = False


def decode_rgb_png(encoded):
    """Read PNG chunks/filter-0 RGB8 rows independently of the renderer."""
    data = base64.b64decode(encoded, validate=True)
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("Invalid PNG signature")
    position, compressed, dimensions = 8, bytearray(), None
    while position < len(data):
        length = struct.unpack_from(">I", data, position)[0]
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + length]
        checksum = struct.unpack_from(">I", data, position + 8 + length)[0]
        if checksum != zlib.crc32(kind + payload) & 0xffffffff:
            raise AssertionError("Invalid PNG chunk checksum")
        if kind == b"IHDR":
            width, height, depth, colour, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            if (depth, colour, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise AssertionError("Expected a non-interlaced RGB8 image")
            dimensions = width, height
        elif kind == b"IDAT":
            compressed.extend(payload)
        position += length + 12
        if kind == b"IEND":
            break
    if dimensions is None or position != len(data):
        raise AssertionError("Incomplete PNG structure")
    width, height = dimensions
    raw = zlib.decompress(compressed)
    stride = 1 + width * 3
    if len(raw) != height * stride:
        raise AssertionError("PNG dimensions do not match decompressed bytes")
    pixels = []
    for row in range(height):
        start = row * stride
        if raw[start] != 0:
            raise AssertionError("The lossless fixture expects filter-0 rows")
        pixels.append([tuple(raw[start + 1 + column * 3:start + 4 + column * 3])
                       for column in range(width)])
    return width, height, pixels


class RecordedFrameFidelityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hipc-view-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.output = self.directory / "rendered"
        self.nx, self.ny, self.dx, self.dy = 16, 8, .25, .125
        self.pitch = self.nx + 2
        self.mask = fluid_mask(self.nx, self.ny, self.dx, self.dy, "2412")
        self.settings = {"nx": self.nx, "ny": self.ny, "airfoil": "2412",
                         "dt": .01, "end_time": .03}
        self.runs = [{"backend": "serial", "status": "completed", "returncode": 0,
                      "portable_command": ["airfoil_serial", "--noio"]}]
        self.records = [self.record(index, amplitude)
                        for index, amplitude in enumerate((4, 2, 1), start=1)]

    def visible(self, field, i, j):
        index = j * self.pitch + i
        return self.mask[index] and (field == "p" or self.mask[index + 1])

    def record(self, cycle, amplitude):
        points = self.pitch * (self.ny + 2)
        velocity, pressure = [999999.0] * (3 * points), [999999.0] * points
        for j in range(1, self.ny + 1):
            for i in range(1, self.nx + 1):
                index = j * self.pitch + i
                sign = (0, 1, -1)[(i + 2 * j) % 3]
                # Huge hidden values must not influence the displayed fluid scale.
                if self.visible("u", i, j):
                    velocity[index * 3:index * 3 + 3] = [sign * amplitude, 7.0, 0.0]
                if self.visible("p", i, j):
                    pressure[index] = sign * 2 * amplitude
        grid = Grid(Path(f"state-{cycle}.vtr"), (0, self.nx + 1, 0, self.ny + 1, 0, 0),
                    ([i * self.dx for i in range(self.nx + 2)],
                     [j * self.dy for j in range(self.ny + 2)], [0.0]),
                    cycle * .01, cycle,
                    {"uv": Field("PointData", 3, velocity), "p": Field("PointData", 1, pressure)}, [])
        return {"grid": grid, "comparison": compare_grids(deepcopy(grid), grid)}

    def render(self, records=None):
        final = self.records[-1]
        return write_demo_page(self.output, final["grid"], final["comparison"],
                               self.runs, self.settings, frames=records or self.records)

    def svg(self, path):
        tree = ET.parse(path)
        return tree, json.loads(tree.find(SVG + "metadata").text)

    def source_values(self, state, field):
        values = state.fields["uv" if field == "u" else "p"].values
        return [values[3 * (j * self.pitch + i)] if field == "u" else values[j * self.pitch + i]
                for j in range(1, self.ny + 1) for i in range(1, self.nx + 1)]

    def assert_metadata(self, metadata, state, field):
        limit = 4 if field == "u" else 8
        self.assertEqual((metadata["time"], metadata["cycle"]), (state.time, state.cycle))
        self.assertEqual((metadata["nx"], metadata["ny"], metadata["dx"], metadata["dy"]),
                         (self.nx, self.ny, self.dx, self.dy))
        self.assertEqual(metadata["colour_limits"], [-limit, limit])
        self.assertEqual(metadata["visible_samples"], sum(self.visible(field, i, j)
                         for j in range(1, self.ny + 1) for i in range(1, self.nx + 1)))
        self.assertEqual(metadata["masked_samples"], self.nx * self.ny - metadata["visible_samples"])
        samples = self.source_values(state, field)
        digest = hashlib.sha256(struct.pack(f">{len(samples)}d", *samples)).hexdigest()
        self.assertEqual(metadata["sample_values_sha256"], digest)
        self.assertEqual(metadata["source_vtr"], state.path.name)

    def test_all_recorded_pixels_metadata_and_views_preserve_source_data(self):
        page = self.render()
        parser = TimelineReader()
        parser.feed(page.read_text())
        timeline = json.loads("".join(parser.parts))
        self.assertEqual([(frame["time"], frame["cycle"]) for frame in timeline],
                         [(record["grid"].time, record["grid"].cycle) for record in self.records])
        files = set()
        positions = {}
        for record, frame in zip(self.records, timeline):
            state = record["grid"]
            self.assertEqual(frame["time_exact"], repr(state.time))
            for field, limit in (("u", 4), ("p", 8)):
                images = []
                for view in ("full", "detail"):
                    path = self.output / frame["files"][field][view]
                    self.assertTrue(path.is_file())
                    files.add(path)
                    tree, metadata = self.svg(path)
                    self.assert_metadata(metadata, state, field)
                    positions.setdefault(field, metadata["sample_positions"])
                    self.assertEqual(metadata["sample_positions"], positions[field])
                    self.assertEqual(metadata["view"], view)
                    image = tree.find(f".//{SVG}image[@class='sample-image']")
                    self.assertIsNotNone(image)
                    bounds, plot = metadata["bounds"], metadata["plot_rect"]
                    scale = plot["width"] / (bounds["xmax"] - bounds["xmin"])
                    # Decode the image transform back to physical coordinates.
                    # Detail/full views must crop the same staggered source grid.
                    self.assertAlmostEqual(bounds["xmin"] + (float(image.get("x")) - plot["left"]) / scale,
                                           self.dx / 2 if field == "u" else 0, places=7)
                    self.assertAlmostEqual(float(image.get("width")) / scale, self.nx * self.dx, places=7)
                    self.assertAlmostEqual(float(image.get("height")) / scale, self.ny * self.dy, places=7)
                    self.assertAlmostEqual(bounds["ymax"] - (float(image.get("y")) +
                                           float(image.get("height")) - plot["top"]) / scale, 0, places=7)
                    prefix, encoded = image.get("href").split(",", 1)
                    self.assertEqual(prefix, "data:image/png;base64")
                    images.append(encoded)
                    width, height, pixels = decode_rgb_png(encoded)
                    self.assertEqual((width, height), (self.nx, self.ny))
                    samples = self.source_values(state, field)
                    for row in range(height):
                        j = self.ny - row
                        for column in range(width):
                            i = column + 1
                            value = samples[(j - 1) * self.nx + i - 1]
                            expected = PALETTE[value / limit] if self.visible(field, i, j) else MASK_RGB
                            self.assertEqual(pixels[row][column], expected,
                                             f"{field}/{view} CYCLE={state.cycle}, i={i}, j={j}")
                self.assertEqual(images[0], images[1], "Zooming must not resample or alter the image")
        self.assertEqual(len(files), 12)
        self.assertNotEqual(positions["u"], positions["p"], "Staggered fields must retain different x locations")

        # The final standalone plots must use the same cross-frame limits, even
        # though their own visible amplitudes are four times smaller.
        state = self.records[-1]["grid"]
        for field, stem in (("u", "velocity"), ("p", "pressure")):
            for suffix in ("", "-detail"):
                tree, metadata = self.svg(self.output / f"{stem}{suffix}.svg")
                self.assert_metadata(metadata, state, field)
                for rectangle in tree.findall(f".//{SVG}rect[@class='sample']"):
                    i, j = int(rectangle.get("data-i")), int(rectangle.get("data-j"))
                    self.assertTrue(self.visible(field, i, j))
                    value = self.source_values(state, field)[(j - 1) * self.nx + i - 1]
                    self.assertEqual(float(rectangle.get("data-value")), value)

    def test_failed_intermediate_comparison_cannot_produce_success_page(self):
        records = deepcopy(self.records)
        changed = records[1]["grid"]
        reference = deepcopy(changed)
        changed.fields["p"].values[self.pitch + 1] += 1
        records[1]["comparison"] = compare_grids(reference, changed)
        self.assertFalse(records[1]["comparison"]["passed"])
        with self.assertRaisesRegex(ValueError, "Every displayed frame must pass"):
            self.render(records)
        self.assertFalse((self.output / "index.html").exists())

    def test_repeated_frame_time_is_rejected_before_page_creation(self):
        records = deepcopy(self.records)
        records[1]["grid"].time = records[0]["grid"].time
        records[1]["comparison"] = compare_grids(deepcopy(records[1]["grid"]), records[1]["grid"])
        with self.assertRaisesRegex(ValueError, "times must increase strictly"):
            self.render(records)
        self.assertFalse((self.output / "index.html").exists())

    def test_changed_intermediate_coordinates_are_rejected(self):
        records = deepcopy(self.records)
        records[1]["grid"].coordinates[0][2] += .01
        records[1]["comparison"] = compare_grids(deepcopy(records[1]["grid"]), records[1]["grid"])
        with self.assertRaisesRegex(ValueError, "same grid geometry"):
            self.render(records)
        self.assertFalse((self.output / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
