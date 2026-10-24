// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <spectre/Exporter.h>

int main(int argc, char** argv) {
  // Parse CLI arguments
  if (argc != 9) {
    fprintf(stderr,
            "Usage: %s VOLUME_FILE SUBFILE_NAME STEP FIELD X Y Z EXPECTED\n",
            argv[0]);
    return 1;
  }
  const char* volume_file = argv[1];
  const char* subfile_name = argv[2];
  const int step = atoi(argv[3]);
  const char* field = argv[4];
  const double x = atof(argv[5]);
  const double y = atof(argv[6]);
  const double z = atof(argv[7]);
  const double expected = atof(argv[8]);
  // Interpolate data
  const double* target_points[3] = {&x, &y, &z};
  double result = NAN;
  double* out[1] = {&result};
  if (spectre_interpolate_to_points(out, &volume_file, 1, subfile_name, step,
                                    &field, 1, target_points, 3, 1, false,
                                    false, 0) != 0) {
    fprintf(stderr, "FAILURE: %s\n", spectre_exporter_last_error());
    return 1;
  }
  // Check that errors are returned to the caller
  if (spectre_interpolate_to_points(out, &volume_file, 1, "nonexistent", step,
                                    &field, 1, target_points, 3, 1, false,
                                    false, 0) == 0 ||
      spectre_exporter_last_error()[0] == '\0') {
    fprintf(stderr, "FAILURE: Expected an error for a nonexistent subfile.\n");
    return 1;
  }
  // Check result
  if (fabs(result - expected) < 1.e-10) {
    printf("SUCCESS\n");
    return 0;
  } else {
    fprintf(stderr, "FAILURE. Result is: %.17g\n", result);
    return 1;
  }
}
