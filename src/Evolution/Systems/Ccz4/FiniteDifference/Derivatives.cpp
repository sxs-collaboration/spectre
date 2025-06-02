// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"

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
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SecondPartialDerivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SecondPartialDerivatives.tpp"
#include "Evolution/Systems/Ccz4/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
void spacetime_derivatives(
    const gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::deriv, typename System::gradients_tags,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        result,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian) {
  using gradients_tags = typename System::gradients_tags;
  if (UNLIKELY(result->number_of_grid_points() !=
               volume_evolved_variables.number_of_grid_points())) {
    result->initialize(volume_evolved_variables.number_of_grid_points());
  }

  using first_Ccz4_tag = tmpl::front<Tags::spacetime_reconstruction_tags>;
  constexpr size_t number_of_Ccz4_components =
      Variables<Tags::spacetime_reconstruction_tags>::
          number_of_independent_components;

  DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [directional_element_id, ghost_data] : all_ghost_data) {
    using NeighborVariables =
        Variables<Tags::spacetime_reconstruction_tags>;
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
        gsl::make_span(get<first_Ccz4_tag>(view)[0].data(),
                       number_of_Ccz4_components * neighbor_number_of_points)});
  }

  const auto volume_Ccz4_vars =
      gsl::make_span(get<first_Ccz4_tag>(volume_evolved_variables)[0].data(),
                     number_of_Ccz4_components *
                         volume_evolved_variables.number_of_grid_points());

  ::fd::partial_derivatives<gradients_tags>(
      result, volume_Ccz4_vars, ghost_cell_vars, volume_mesh,
      number_of_Ccz4_components, deriv_order,
      cell_centered_logical_to_inertial_inv_jacobian);
}

void second_spacetime_derivatives(
    const gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::second_deriv, System::gradients_tags,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        result,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian) {
  ASSERT(deriv_order == 4,
         "Only fourth order second partial derivatives have been implemented.");

  using gradients_tags = typename System::gradients_tags;
  if (UNLIKELY(result->number_of_grid_points() !=
               volume_evolved_variables.number_of_grid_points())) {
    result->initialize(volume_evolved_variables.number_of_grid_points());
  }

  using first_Ccz4_tag = tmpl::front<Tags::spacetime_reconstruction_tags>;
  constexpr size_t number_of_Ccz4_components =
      Variables<Tags::spacetime_reconstruction_tags>::
          number_of_independent_components;

  DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [directional_element_id, ghost_data] : all_ghost_data) {
    using NeighborVariables =
        Variables<Tags::spacetime_reconstruction_tags>;
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
        gsl::make_span(get<first_Ccz4_tag>(view)[0].data(),
                       number_of_Ccz4_components * neighbor_number_of_points)});
  }

  const auto volume_Ccz4_vars =
      gsl::make_span(get<first_Ccz4_tag>(volume_evolved_variables)[0].data(),
                     number_of_Ccz4_components *
                         volume_evolved_variables.number_of_grid_points());

  second_partial_derivatives<gradients_tags>(
      result, volume_Ccz4_vars, ghost_cell_vars, volume_mesh,
      number_of_Ccz4_components, deriv_order,
      cell_centered_logical_to_inertial_inv_jacobian);
}
}  // namespace Ccz4::fd
