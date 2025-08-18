// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::GhValenciaDivClean::fd {
/*!
 * \brief Helper function that takes spacetime variable data from a
 * `DirectionalIdMap` containing all neighbor data and copies them to
 * `ghost_cell_spacetime_vars`, a separate `DirectionMap`.
 *
 * \tparam NeighborVariables a `Variables` type containing all tags stored in
 * the neighbor data
 * \tparam FirstGhTag the first spacetime variables tag in
 * `NeighborVariables`. Note: it is assumed that the spacetime variables are
 * consecutively stored in the `Variables`
 * \param ghost_cell_spacetime_vars a `DirectionMap` to be filled with spans
 * storing the spacetime data
 * \param all_ghost_data a `DirectionalIdMap` containing all neighbor data
 * \param number_of_gh_components the number of independent components for all
 * spacetime variables in `NeighborVariables`
 */
template <typename NeighborVariables, typename FirstGhTag>
void fill_neighbor_spacetime_variables(
    const gsl::not_null<DirectionMap<3, gsl::span<const double>>*>
        ghost_cell_spacetime_vars,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t number_of_gh_components) {
  for (const auto& [directional_element_id, ghost_data] : all_ghost_data) {
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
    // Note: assumes that the spacetime tags are consecutive in
    // `NeighborVariables`
    ghost_cell_spacetime_vars->insert(std::pair{
        directional_element_id.direction(),
        gsl::make_span(get<FirstGhTag>(view)[0].data(),
                       number_of_gh_components * neighbor_number_of_points)});
  }
}

}  // namespace grmhd::GhValenciaDivClean::fd
