// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/FillNeighborSpacetimeVariables.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd."
    "FillNeighborSpacetimeVariables",
    "[Unit][Evolution]") {
  const size_t points_per_dimension = 5;
  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};

  using NeighborVariables =
      Variables<grmhd::GhValenciaDivClean::Tags::
                    primitive_grmhd_and_spacetime_reconstruction_tags>;
  using gh_tags =
      grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags;
  constexpr size_t number_of_gh_components = Variables<
      grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>::
      number_of_independent_components;

  DirectionalIdMap<3, evolution::dg::subcell::GhostData> neighbor_data{};
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  for (const auto& direction : Direction<3>::all_directions()) {
    const auto neighbor_vars = make_with_random_values<NeighborVariables>(
        make_not_null(&gen), make_not_null(&dist),
        subcell_mesh.number_of_grid_points());

    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars), subcell_mesh.extents(), 2,
        std::unordered_set{direction.opposite()}, 0, {});
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    const auto key = DirectionalId<3>{direction, ElementId<3>{0}};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }

  DirectionMap<3, gsl::span<const double>> result{};
  grmhd::GhValenciaDivClean::fd::fill_neighbor_spacetime_variables<
      NeighborVariables, tmpl::front<gh_tags>>(
      make_not_null(&result), neighbor_data, number_of_gh_components);

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto key = DirectionalId<3>{direction, ElementId<3>{0}};
    auto& result_in_dir = result[direction];
    auto& ghost_data_in_dir =
        (neighbor_data[key].neighbor_ghost_data_for_reconstruction());
    const size_t neighbor_number_of_points =
        ghost_data_in_dir.size() /
        NeighborVariables::number_of_independent_components;

    const NeighborVariables
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        view{const_cast<double*>(ghost_data_in_dir.data()),
             neighbor_number_of_points *
                 NeighborVariables::number_of_independent_components};
    const auto subset_view = view.extract_subset<gh_tags>();
    const auto compare = subset_view.data();
    for (size_t element = 0;
         element < number_of_gh_components * neighbor_number_of_points;
         ++element) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      CHECK(result_in_dir[element] == *(compare + element));
    };
  }
}
}  // namespace
