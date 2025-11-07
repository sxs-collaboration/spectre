// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/Filter.hpp"

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
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/Filter.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
void ccz4_kreiss_oliger_filter(
    const gsl::not_null<Variables<System::variables_tag_list>*> result,
    const Variables<System::variables_tag_list>& volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const bool evolve_lapse_and_shift, const Mesh<3>& volume_mesh,
    const size_t order, const double epsilon) {
  // The result is assumed to be initialized to the volume_evolved_variables
  ASSERT(result->number_of_grid_points() ==
             volume_evolved_variables.number_of_grid_points(),
         "The result and volume_evolved_variables must have the same number of "
         "grid points. Found "
             << result->number_of_grid_points() << " and "
             << volume_evolved_variables.number_of_grid_points());

  using first_ccz4_tag = tmpl::front<System::variables_tag_list>;
  const size_t number_of_ccz4_components =
      evolve_lapse_and_shift
          ? Variables<
                System::variables_tag_list>::number_of_independent_components
          : Variables<
                System::variables_tag_list>::number_of_independent_components -
                7;

  DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [directional_element_id, ghost_data] : all_ghost_data) {
    using NeighborVariables = Variables<System::variables_tag_list>;
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
        gsl::make_span(get<first_ccz4_tag>(view)[0].data(),
                       number_of_ccz4_components * neighbor_number_of_points)});
  }

  const auto volume_ccz4_vars =
      gsl::make_span(get<first_ccz4_tag>(volume_evolved_variables)[0].data(),
                     number_of_ccz4_components *
                         volume_evolved_variables.number_of_grid_points());

  auto filtered_ccz4_vars =
      gsl::make_span(get<first_ccz4_tag>(*result)[0].data(),
                     number_of_ccz4_components *
                         volume_evolved_variables.number_of_grid_points());
  ::fd::kreiss_oliger_filter(make_not_null(&filtered_ccz4_vars),
                             volume_ccz4_vars, ghost_cell_vars, volume_mesh,
                             number_of_ccz4_components, order, epsilon);
}

}  // namespace Ccz4::fd
