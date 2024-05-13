// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Filters.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/Filter.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::fd {
void spacetime_kreiss_oliger_filter(
    const gsl::not_null<Variables<
        typename grmhd::GhValenciaDivClean::System::variables_tag::tags_list>*>
        result,
    const Variables<
        typename grmhd::GhValenciaDivClean::System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const Mesh<3>& volume_mesh, const size_t order, const double epsilon) {
  if (UNLIKELY(result->number_of_grid_points() !=
               volume_evolved_variables.number_of_grid_points())) {
    result->initialize(volume_evolved_variables.number_of_grid_points());
  }

  using first_gh_tag = tmpl::front<
      grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>;
  constexpr size_t number_of_gh_components = Variables<
      grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>::
      number_of_independent_components;

  DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [directional_element_id, ghost_data] : all_ghost_data) {
    using NeighborVariables =
        Variables<grmhd::GhValenciaDivClean::Tags::
                      primitive_grmhd_and_spacetime_reconstruction_tags>;
    const DataVector& neighbor_data =
        ghost_data.neighbor_ghost_data_for_reconstruction();
    const size_t neighbor_number_of_points =
        neighbor_data.size() /
        NeighborVariables::number_of_independent_components;
    ASSERT(
        neighbor_data.size() %
                NeighborVariables::number_of_independent_components ==
            0,
        "Amount of reconstruction data sent ("
            << neighbor_data.size() << ") from " << directional_element_id
            << " is not a multiple of the number of reconstruction variables "
            << NeighborVariables::number_of_independent_components);
    // Use a Variables view to get offset into spacetime variables
    // without having to do pointer math.
    const NeighborVariables
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        view{const_cast<double*>(neighbor_data.data()),
             neighbor_number_of_points *
                 NeighborVariables::number_of_independent_components};
    ghost_cell_vars.insert(std::pair{
        directional_element_id.direction(),
        gsl::make_span(get<first_gh_tag>(view)[0].data(),
                       number_of_gh_components * neighbor_number_of_points)});
  }

  const auto volume_gh_vars =
      gsl::make_span(get<first_gh_tag>(volume_evolved_variables)[0].data(),
                     number_of_gh_components *
                         volume_evolved_variables.number_of_grid_points());

  auto filtered_gh_vars =
      gsl::make_span(get<first_gh_tag>(*result)[0].data(),
                     number_of_gh_components *
                         volume_evolved_variables.number_of_grid_points());
  ::fd::kreiss_oliger_filter(make_not_null(&filtered_gh_vars), volume_gh_vars,
                             ghost_cell_vars, volume_mesh,
                             number_of_gh_components, order, epsilon);
}
}  // namespace grmhd::GhValenciaDivClean::fd
