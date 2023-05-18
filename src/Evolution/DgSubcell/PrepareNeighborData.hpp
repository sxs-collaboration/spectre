// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Tags/NeighborMesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
/*!
 * \brief Add local data for our and our neighbor's relaxed discrete maximum
 * principle troubled-cell indicator, and for reconstruction.
 *
 * The local maximum and minimum of the evolved variables is added to
 * `Tags::DataForRdmpTci` for the RDMP TCI. Then the
 * data needed by neighbor elements to do reconstruction on the FD grid is sent.
 * The data to be sent is computed in the mutator
 * `Metavariables::SubcellOptions::GhostVariables`, which returns a
 * `Variables` of the tensors to send to the neighbors. The main reason for
 * having the mutator `GhostVariables` is to allow sending primitive or
 * characteristic variables for reconstruction.
 *
 * \note If all neighbors are using DG then we send our DG volume data _without_
 * orienting it. This elides the expense of projection and slicing. If any
 * neighbors are doing FD, we project and slice to all neighbors. A future
 * optimization would be to measure the cost of slicing data, and figure out how
 * many neighbors need to be doing FD before it's worth projecting and slicing
 * vs. just projecting to the ghost cells. Another optimization is to always
 * send DG volume data to neighbors that we think are doing DG, so that we can
 * elide any slicing or projection cost in that direction.
 */
template <typename Metavariables, typename DbTagsList, size_t Dim>
void prepare_neighbor_data(
    const gsl::not_null<DirectionMap<Dim, DataVector>*>
        all_neighbor_data_for_reconstruction,
    const gsl::not_null<Mesh<Dim>*> ghost_data_mesh,
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    [[maybe_unused]] const Variables<db::wrap_tags_in<
        ::Tags::Flux, typename Metavariables::system::flux_variables,
        tmpl::size_t<Dim>, Frame::Inertial>>& volume_fluxes) {
  const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(*box);
  const Mesh<Dim>& subcell_mesh =
      db::get<::evolution::dg::subcell::Tags::Mesh<Dim>>(*box);
  const auto& element = db::get<::domain::Tags::Element<Dim>>(*box);

  const std::unordered_set<Direction<Dim>>& directions_to_slice =
      element.internal_boundaries();

  const auto& neighbor_meshes =
      db::get<evolution::dg::Tags::NeighborMesh<Dim>>(*box);
  const subcell::RdmpTciData& rdmp_data =
      db::get<subcell::Tags::DataForRdmpTci>(*box);
  const ::fd::DerivativeOrder fd_derivative_order =
      get<evolution::dg::subcell::Tags::SubcellOptions<Dim>>(*box)
          .finite_difference_derivative_order();
  const size_t rdmp_size = rdmp_data.max_variables_values.size() +
                           rdmp_data.min_variables_values.size();
  const size_t extra_size_for_ghost_data =
      (fd_derivative_order == ::fd::DerivativeOrder::Two
           ? 0
           : volume_fluxes.size());

  if (DataVector ghost_variables =
          [&box, &extra_size_for_ghost_data, &rdmp_size, &volume_fluxes]() {
            if (extra_size_for_ghost_data == 0) {
              return db::mutate_apply(
                  typename Metavariables::SubcellOptions::GhostVariables{}, box,
                  rdmp_size);
            } else {
              DataVector ghost_vars = db::mutate_apply(
                  typename Metavariables::SubcellOptions::GhostVariables{}, box,
                  rdmp_size + extra_size_for_ghost_data);
              std::copy(
                  volume_fluxes.data(),
                  std::next(volume_fluxes.data(),
                            static_cast<std::ptrdiff_t>(volume_fluxes.size())),
                  std::prev(ghost_vars.end(),
                            static_cast<std::ptrdiff_t>(
                                rdmp_size + extra_size_for_ghost_data)));
              return ghost_vars;
            }
          }();
      alg::all_of(neighbor_meshes,
                  [](const auto& directional_element_id_and_mesh) {
                    ASSERT(directional_element_id_and_mesh.second.basis(0) !=
                               Spectral::Basis::Chebyshev,
                           "Don't yet support Chebyshev basis with DG-FD");
                    return directional_element_id_and_mesh.second.basis(0) ==
                           Spectral::Basis::Legendre;
                  })) {
    *ghost_data_mesh = dg_mesh;
    const size_t total_to_slice = directions_to_slice.size();
    size_t slice_count = 0;
    for (const auto& direction : directions_to_slice) {
      ++slice_count;
      // Move instead of copy on the last iteration. Elides a memory
      // allocation and copy.
      [[maybe_unused]] const auto insert_result =
          UNLIKELY(slice_count == total_to_slice)
              ? all_neighbor_data_for_reconstruction->insert_or_assign(
                    direction, std::move(ghost_variables))
              : all_neighbor_data_for_reconstruction->insert_or_assign(
                    direction, ghost_variables);
      ASSERT(all_neighbor_data_for_reconstruction->size() == total_to_slice or
                 insert_result.second,
             "Failed to insert the neighbor data in direction " << direction);
    }
  } else {
    *ghost_data_mesh = subcell_mesh;
    const size_t ghost_zone_size =
        Metavariables::SubcellOptions::ghost_zone_size(*box);

    const DataVector data_to_project{};
    make_const_view(make_not_null(&data_to_project), ghost_variables, 0,
                    ghost_variables.size() - rdmp_size);
    const DataVector projected_data = evolution::dg::subcell::fd::project(
        data_to_project, dg_mesh, subcell_mesh.extents());
    *all_neighbor_data_for_reconstruction = subcell::slice_data(
        projected_data, db::get<subcell::Tags::Mesh<Dim>>(*box).extents(),
        ghost_zone_size, directions_to_slice, rdmp_size);

    const auto& subcell_options = db::get<Tags::SubcellOptions<Dim>>(*box);

    const bool bordering_dg_block = alg::any_of(
        element.neighbors(),
        [&subcell_options](const auto& direction_and_neighbor) {
          const size_t first_block_id =
              direction_and_neighbor.second.ids().begin()->block_id();
          return alg::found(subcell_options.only_dg_block_ids(),
                            first_block_id);
        });

    // Subcell allowed if not in a DG only block, and not bordering
    // a DG only block.
    const bool subcell_allowed_in_element =
        not std::binary_search(subcell_options.only_dg_block_ids().begin(),
                               subcell_options.only_dg_block_ids().end(),
                               element.id().block_id()) and
        not bordering_dg_block;

    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      // Note: this currently orients the data for _each_ neighbor. We can
      // instead just orient the data once for all in a particular direction.

      // Only subcell elements check for multiple neighbors
      ASSERT(not subcell_allowed_in_element or neighbors.ids().size() == 1,
             "Subcell elements can only have one neighbor per direction "
             "right now.");

      if (not orientation.is_aligned()) {
        std::array<size_t, Dim> slice_extents{};
        for (size_t d = 0; d < Dim; ++d) {
          gsl::at(slice_extents, d) = subcell_mesh.extents(d);
        }

        gsl::at(slice_extents, direction.dimension()) = ghost_zone_size;

        // Only hash the direction once.
        auto& neighbor_data =
            all_neighbor_data_for_reconstruction->at(direction);
        // Make a copy of the local orientation data, then we can re-orient
        // directly into the send buffer.
        DataVector local_orientation_data(neighbor_data.size() - rdmp_size);
        std::copy(neighbor_data.begin(),
                  std::prev(neighbor_data.end(), static_cast<int>(rdmp_size)),
                  local_orientation_data.data());
        // Orient variables expects only the neighbor data, not neighbor data +
        // rdmp data, so create a non-owning DataVector to just the neighbor
        // data
        DataVector oriented_data{neighbor_data.data(),
                                 neighbor_data.size() - rdmp_size};
        orient_variables(make_not_null(&oriented_data), local_orientation_data,
                         Index<Dim>{slice_extents}, orientation);
      }
    }
  }

  // Note: The RDMP TCI data must be filled by the TCIs before getting to this
  // call. That means once in the initial data and then in both the DG and FD
  // TCIs.
  for (const auto& [direction, neighbors] : element.neighbors()) {
    // Add the RDMP TCI data to what we will be sending.
    // Note that this is added _after_ the reconstruction data has been
    // re-oriented (in the case where we have neighbors doing FD).
    DataVector& neighbor_data =
        all_neighbor_data_for_reconstruction->at(direction);
    std::copy(rdmp_data.max_variables_values.begin(),
              rdmp_data.max_variables_values.end(),
              std::prev(neighbor_data.end(), static_cast<int>(rdmp_size)));
    std::copy(
        rdmp_data.min_variables_values.begin(),
        rdmp_data.min_variables_values.end(),
        std::prev(neighbor_data.end(),
                  static_cast<int>(rdmp_data.min_variables_values.size())));
  }
}
}  // namespace evolution::dg::subcell
