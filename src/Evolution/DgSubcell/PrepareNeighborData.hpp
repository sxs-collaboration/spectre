// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
/*!
 * \brief Add local data for our and our neighbor's relaxed discrete maximum
 * principle troubled-cell indicator, and compute and slice data needed for the
 * neighbor cells.
 *
 * The local maximum and minimum of the evolved variables is added to
 * `Tags::NeighborDataForReconstructionAndRdmpTci` for the RDMP TCI. Then the
 * data needed by neighbor elements to do reconstruction on the FD grid is sent.
 * The data to be sent is computed in the mutator
 * `Metavariables::SubcellOptions::GhostDataToSlice`, which returns a
 * `Variables` of the tensors to slice and send to the neighbors. Note that the
 * `Tags::Inactive<variables_tag>` are already projected onto the subcells
 * before this function is called. This is because the projection needs to be
 * done anyway for the a posteriori TCI. That is, the evolved variables are
 * projected at the end of the time step rather than the beginning, while this
 * function is called at the beginning. The main reason for having the mutator
 * `GhostDataToSlice` is to allow sending primitive or characteristic variables
 * for reconstruction.
 */
template <typename Metavariables, typename DbTagsList>
DirectionMap<Metavariables::volume_dim, std::vector<double>>
prepare_neighbor_data(const gsl::not_null<db::DataBox<DbTagsList>*> box) {
  constexpr size_t volume_dim = Metavariables::volume_dim;
  using variables_tag = typename Metavariables::system::variables_tag;
  const std::pair self_id{Direction<volume_dim>::lower_xi(),
                          ElementId<volume_dim>::external_boundary_id()};
  db::mutate<
      subcell::Tags::NeighborDataForReconstructionAndRdmpTci<volume_dim>>(
      box,
      [&self_id](
          const auto neighbor_data_ptr,
          std::pair<std::vector<double>, std::vector<double>> max_min_of_vars) {
        subcell::NeighborData rdmp_data{};
        rdmp_data.max_variables_values = std::move(max_min_of_vars.first);
        rdmp_data.min_variables_values = std::move(max_min_of_vars.second);
        (*neighbor_data_ptr)[self_id] = std::move(rdmp_data);
      },
      evolution::dg::subcell::rdmp_max_min(
          db::get<variables_tag>(*box),
          db::get<evolution::dg::subcell::Tags::Inactive<variables_tag>>(*box),
          true));

  const auto& element = db::get<::domain::Tags::Element<volume_dim>>(*box);
  DirectionMap<volume_dim, std::vector<double>>
      all_neighbor_data_for_reconstruction{};

  DirectionMap<volume_dim, bool> directions_to_slice{};
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    if (neighbors_in_direction.size() == 0) {
      directions_to_slice[direction] = false;
    } else {
      directions_to_slice[direction] = true;
    }
  }

  const size_t ghost_zone_size =
      Metavariables::SubcellOptions::ghost_zone_size(*box);

  // We can assume that the inactive tags are projected because if the TCI
  // passed at the end of the last step, then as part of the RDMP it
  // projected the data. We require the data be projected during
  // initialization.
  all_neighbor_data_for_reconstruction = subcell::slice_data(
      db::mutate_apply(
          typename Metavariables::SubcellOptions::GhostDataToSlice{}, box),
      db::get<subcell::Tags::Mesh<volume_dim>>(*box).extents(), ghost_zone_size,
      directions_to_slice);

  ASSERT(
      db::get<
          subcell::Tags::NeighborDataForReconstructionAndRdmpTci<volume_dim>>(
          *box)
          .contains(self_id),
      "The self info for the RDMP TCI should have been added to the neighbor "
      "data tag but wasn't.");
  const subcell::NeighborData& rdmp_data =
      db::get<
          subcell::Tags::NeighborDataForReconstructionAndRdmpTci<volume_dim>>(
          *box)
          .at(self_id);

  for (const auto& [direction, neighbors] : element.neighbors()) {
    const auto& orientation = neighbors.orientation();
    const Mesh<volume_dim>& subcell_mesh =
        db::get<subcell::Tags::Mesh<volume_dim>>(*box);
    if (not orientation.is_aligned()) {
      std::array<size_t, volume_dim> slice_extents{};
      for (size_t d = 0; d < volume_dim; ++d) {
        gsl::at(slice_extents, d) = subcell_mesh.extents(d);
      }

      gsl::at(slice_extents, direction.dimension()) = ghost_zone_size;

      all_neighbor_data_for_reconstruction.at(direction) =
          orient_variables(all_neighbor_data_for_reconstruction.at(direction),
                           Index<volume_dim>{slice_extents}, orientation);
    }

    // Add the RDMP TCI data to what we will be sending.
    // Note that this is added _after_ the reconstruction data has been
    // re-oriented.
    std::vector<double>& neighbor_subcell_data =
        all_neighbor_data_for_reconstruction.at(direction);
    neighbor_subcell_data.reserve(neighbor_subcell_data.size() +
                                  rdmp_data.max_variables_values.size() +
                                  rdmp_data.min_variables_values.size());
    neighbor_subcell_data.insert(neighbor_subcell_data.end(),
                                 rdmp_data.max_variables_values.begin(),
                                 rdmp_data.max_variables_values.end());
    neighbor_subcell_data.insert(neighbor_subcell_data.end(),
                                 rdmp_data.min_variables_values.begin(),
                                 rdmp_data.min_variables_values.end());
  }

  return all_neighbor_data_for_reconstruction;
}
}  // namespace evolution::dg::subcell
