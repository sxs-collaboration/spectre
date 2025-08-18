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
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {
struct Metavariables {
  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = void;
    using projectors = tmpl::list<>;
    static constexpr bool keep_coarse_grids = false;
    [[maybe_unused]] static constexpr bool p_refine_only_in_event = false;
  };
};

template <size_t Dim>
void test() {
  auto box = db::create<
      db::AddSimpleTags<amr::Tags::Info<Dim>, amr::Tags::NeighborInfo<Dim>>>(
      amr::Info<Dim>{}, std::unordered_map<ElementId<Dim>, amr::Info<Dim>>{});
  db::mutate_apply<amr::Initialization::Initialize<Dim, Metavariables>>(
      make_not_null(&box));
  CHECK(db::get<amr::Tags::Info<Dim>>(box).flags ==
        make_array<Dim>(amr::Flag::Undefined));
  CHECK(db::get<amr::Tags::Info<Dim>>(box).new_mesh == Mesh<Dim>{});
  CHECK(db::get<amr::Tags::NeighborInfo<Dim>>(box).empty());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAmr.Actions.Initialize",
                  "[ParallelAlgorithms][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
