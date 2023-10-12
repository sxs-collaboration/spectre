// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "Utilities/GetOutput.hpp"

namespace ah {
SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Destination",
                  "[ApparentHorizonFinder][Unit]") {
  CHECK(get_output(Destination::ControlSystem) == "ControlSystem");
  CHECK(get_output(Destination::Observation) == "Observation");
  CHECK(TestHelpers::test_creation<Destination>("ControlSystem") ==
        Destination::ControlSystem);
  CHECK(TestHelpers::test_creation<Destination>("Observation") ==
        Destination::Observation);
}
}  // namespace ah
