// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "IO/Observer/TypeOfObservation.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.IO.Observers.TypeOfObservation", "[Unit][Observers]") {
  using observers::TypeOfObservation;
  CHECK(get_output(TypeOfObservation::Reduction) == "Reduction");
  CHECK(get_output(TypeOfObservation::Volume) == "Volume");
  CHECK(get_output(TypeOfObservation::ReductionAndVolume) ==
        "ReductionAndVolume");
}
