// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/InitializeCurrentIteration.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Framework/TestingFramework.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CSW.Worldtube.InitializeCurrentIteration",
    "[Unit][Evolution]") {
  const size_t current_iteration = 77;
  auto box =
      db::create<db::AddSimpleTags<Tags::CurrentIteration>>(current_iteration);
  db::mutate_apply<Initialization::InitializeCurrentIteration>(
      make_not_null(&box));
  CHECK(get<Tags::CurrentIteration>(box) == 0);
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
