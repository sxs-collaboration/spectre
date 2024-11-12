// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.EventsAndTriggers.WhenToCheck",
                  "[Unit][ParallelAlgorithms]") {
  CHECK(get_output(Triggers::WhenToCheck::AtIterations) == "AtIterations");
  CHECK(get_output(Triggers::WhenToCheck::AtSlabs) == "AtSlabs");
  CHECK(get_output(Triggers::WhenToCheck::AtSteps) == "AtSteps");
}
