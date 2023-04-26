// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestHelpers.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"

SPECTRE_TEST_CASE("Unit.Time.AdaptiveSteppingDiagnostics", "[Unit][Time]") {
  const AdaptiveSteppingDiagnostics diags{1, 2, 3, 4, 5};
  CHECK(diags.number_of_slabs == 1);
  CHECK(diags.number_of_slab_size_changes == 2);
  CHECK(diags.number_of_steps == 3);
  CHECK(diags.number_of_step_fraction_changes == 4);
  CHECK(diags.number_of_step_rejections == 5);

  CHECK(diags == AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});
  CHECK(diags != AdaptiveSteppingDiagnostics{2, 2, 3, 4, 5});
  CHECK(diags != AdaptiveSteppingDiagnostics{1, 3, 3, 4, 5});
  CHECK(diags != AdaptiveSteppingDiagnostics{1, 2, 4, 4, 5});
  CHECK(diags != AdaptiveSteppingDiagnostics{1, 2, 3, 5, 5});
  CHECK(diags != AdaptiveSteppingDiagnostics{1, 2, 3, 4, 6});
  CHECK_FALSE(diags != AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});
  CHECK_FALSE(diags == AdaptiveSteppingDiagnostics{2, 2, 3, 4, 5});
  CHECK_FALSE(diags == AdaptiveSteppingDiagnostics{1, 3, 3, 4, 5});
  CHECK_FALSE(diags == AdaptiveSteppingDiagnostics{1, 2, 4, 4, 5});
  CHECK_FALSE(diags == AdaptiveSteppingDiagnostics{1, 2, 3, 5, 5});
  CHECK_FALSE(diags == AdaptiveSteppingDiagnostics{1, 2, 3, 4, 6});

  CHECK(diags == serialize_and_deserialize(diags));
}
