// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "ParallelAlgorithms/Amr/Actions/Initialize.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
template <size_t Dim>
void test() {
  auto box = db::create<
      db::AddSimpleTags<amr::Tags::Flags<Dim>, amr::Tags::NeighborFlags<Dim>>>(
      std::array<amr::Flag, Dim>{},
      std::unordered_map<ElementId<Dim>, std::array<amr::Flag, Dim>>{});
  db::mutate_apply<amr::Initialization::Initialize<Dim>>(make_not_null(&box));
  CHECK(db::get<amr::Tags::Flags<Dim>>(box) ==
        make_array<Dim>(amr::Flag::Undefined));
  CHECK(db::get<amr::Tags::NeighborFlags<Dim>>(box).empty());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAmr.Actions.Initialize",
                  "[ParallelAlgorithms][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
