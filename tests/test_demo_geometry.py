"""Compare the reconstructed mask with flags produced by the real C setup."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from demo_geometry import fluid_mask


C_PROBE = r'''
#include <stdio.h>
#include <stdlib.h>
#include "data.h"
#include "setup.h"

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    imax = atoi(argv[1]);
    jmax = atoi(argv[2]);
    airfoil = atoi(argv[3]);
    xlength = imax * strtod(argv[4], NULL);
    ylength = jmax * strtod(argv[5], NULL);
    setup();
    allocate_arrays();
    problem_set_up();
    for (int j = 0; j < jmax + 2; j++)
        for (int i = 0; i < imax + 2; i++)
            putchar((flag[i][j] & C_F) ? '1' : '0');
    putchar('\n');
    free_arrays();
    return 0;
}
'''


class SolverGeometryCrossChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("C compiler cc is unavailable")
        temporary = tempfile.TemporaryDirectory(prefix="hipc-geometry-probe-")
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        source = directory / "probe.c"
        source.write_text(C_PROBE)
        cls.probe = directory / "probe"
        command = [compiler, "-std=c11", "-O0", "-I", str(ROOT / "serial"),
                   str(source), str(ROOT / "serial/setup.c"), str(ROOT / "common/data.c"),
                   str(ROOT / "serial/boundary.c"), str(ROOT / "common/vtk.c"),
                   "-lm", "-o", str(cls.probe)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if run.returncode != 0:
            raise AssertionError(f"Real solver geometry probe did not compile:\n{run.stdout}\n{run.stderr}")

    def test_mask_matches_actual_c_flags_across_resolutions_and_airfoils(self):
        cases = [
            (128, 32, "2412"), (256, 64, "2412"), (512, 128, "2412"),
            (55, 17, "2412"), (128, 32, "0012"), (55, 17, "0012"),
            (256, 64, "4415"), (55, 17, "4415"), (128, 32, "9999"),
        ]
        for nx, ny, airfoil in cases:
            with self.subTest(nx=nx, ny=ny, airfoil=airfoil):
                dx, dy = 4.0 / nx, 1.0 / ny
                run = subprocess.run([str(self.probe), str(nx), str(ny), airfoil,
                                      repr(dx), repr(dy)],
                                     capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                raw = run.stdout.strip()
                self.assertEqual(len(raw), (nx + 2) * (ny + 2))
                self.assertLessEqual(set(raw), {"0", "1"})
                actual = [value == "1" for value in raw]
                reconstructed = fluid_mask(nx, ny, dx, dy, airfoil)
                self.assertEqual(reconstructed, actual,
                                 "Python mask must equal real flag&C_F at every index")
                self.assertTrue(any(reconstructed), "The fixture must contain fluid cells")
                self.assertTrue(any(not reconstructed[j * (nx + 2) + i]
                                    for j in range(1, ny + 1) for i in range(1, nx + 1)),
                                "The fixture must exercise an actual obstacle")

    def test_probe_also_checks_nondefault_domain_lengths(self):
        nx, ny, airfoil = 55, 17, "4415"
        dx, dy = 6.25 / nx, 1.75 / ny
        run = subprocess.run([str(self.probe), str(nx), str(ny), airfoil, repr(dx), repr(dy)],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(fluid_mask(nx, ny, dx, dy, airfoil),
                         [value == "1" for value in run.stdout.strip()])


class GeometryInputTests(unittest.TestCase):
    def test_ghosts_are_nonfluid_and_integer_airfoil_preserves_leading_zeroes(self):
        nx, ny = 55, 17
        mask = fluid_mask(nx, ny, 4 / nx, 1 / ny, 12)
        self.assertEqual(mask, fluid_mask(nx, ny, 4 / nx, 1 / ny, "0012"))
        pitch = nx + 2
        self.assertEqual(len(mask), pitch * (ny + 2))
        self.assertTrue(all(type(value) is bool for value in mask))
        self.assertFalse(any(mask[:pitch]))
        self.assertFalse(any(mask[-pitch:]))
        self.assertTrue(all(not mask[j * pitch] and not mask[j * pitch + nx + 1]
                            for j in range(ny + 2)))

    def test_invalid_geometry_parameters_fail_before_rendering(self):
        for nx, ny, dx, dy, airfoil in [
            (0, 8, .1, .1, "2412"), (8, -1, .1, .1, "2412"),
            (True, 8, .1, .1, "2412"), (8, 8, 0, .1, "2412"),
            (8, 8, .1, float("nan"), "2412"), (8, 8, float("inf"), .1, "2412"),
            (8, 8, .1, .1, "12"), (8, 8, .1, .1, "２４１２"),
            (8, 8, .1, .1, "1012"), (8, 8, .1, .1, -12),
        ]:
            with self.subTest(parameters=(nx, ny, dx, dy, airfoil)):
                with self.assertRaises(ValueError):
                    fluid_mask(nx, ny, dx, dy, airfoil)


if __name__ == "__main__":
    unittest.main()
