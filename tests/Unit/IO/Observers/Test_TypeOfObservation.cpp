// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "IO/Observer/TypeOfObservation.hpp"
#include "Utilities/GetOutput.hpp"

namespace {
enum class Phases0 { A, B, RegisterWithObserver };
enum class Phases1 { A, B, C };

static_assert(observers::has_register_with_observer_v<Phases0>,
              "Failed testing observers::has_register_with_observers");
static_assert(not observers::has_register_with_observer_v<Phases1>,
              "Failed testing observers::has_register_with_observers");
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Observers.TypeOfObservation", "[Unit][Observers]") {
  using observers::TypeOfObservation;
  CHECK(get_output(TypeOfObservation::Reduction) == "Reduction");
  CHECK(get_output(TypeOfObservation::Volume) == "Volume");
  CHECK(get_output(TypeOfObservation::ReductionAndVolume) ==
        "ReductionAndVolume");
}
