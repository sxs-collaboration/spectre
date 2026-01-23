// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.TimeSteppingPolicy", "[Unit][Evolution]") {
  CHECK(get_output(evolution::dg::TimeSteppingPolicy::Uninitialized) ==
        "Uninitialized");
  CHECK(get_output(evolution::dg::TimeSteppingPolicy::EqualRate) ==
        "EqualRate");
  CHECK(get_output(evolution::dg::TimeSteppingPolicy::Conservative) ==
        "Conservative");
}
