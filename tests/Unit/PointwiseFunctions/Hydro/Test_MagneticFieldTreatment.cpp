// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "PointwiseFunctions/Hydro/MagneticFieldTreatment.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <hydro::MagneticFieldTreatment MagFieldTreatment>
void test_construct_from_options() {
  const auto created =
      TestHelpers::test_creation<hydro::MagneticFieldTreatment>(
          get_output(MagFieldTreatment));
  CHECK(created == MagFieldTreatment);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.MagneticFieldTreatment",
                  "[Hydro][Unit]") {
  using hydro::MagneticFieldTreatment;
  CHECK(get_output(MagneticFieldTreatment::AssumeZero) == "AssumeZero");
  CHECK(get_output(MagneticFieldTreatment::AssumeNonZero) == "AssumeNonZero");
  CHECK(get_output(MagneticFieldTreatment::CheckIfZero) == "CheckIfZero");

  test_construct_from_options<MagneticFieldTreatment::AssumeZero>();
  test_construct_from_options<MagneticFieldTreatment::AssumeNonZero>();
  test_construct_from_options<MagneticFieldTreatment::CheckIfZero>();
}
