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
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Reconstructor.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
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

  const auto& neighbor_meshes = db::get<domain::Tags::NeighborMesh<Dim>>(*box);
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
        db::get<evolution::dg::subcell::Tags::Reconstructor>(*box)
            .ghost_zone_size();

    const DataVector data_to_project{};
    make_const_view(make_not_null(&data_to_project), ghost_variables, 0,
                    ghost_variables.size() - rdmp_size);
    const DataVector projected_data = evolution::dg::subcell::fd::project(
        data_to_project, dg_mesh, subcell_mesh.extents());
    typename evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<
        Dim>::type directions_not_to_slice{};
    for (const auto& [directional_element_id, _] :
         db::get<evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<
             Dim>>(*box)) {
      directions_not_to_slice[directional_element_id] = std::nullopt;
    }
    *all_neighbor_data_for_reconstruction = subcell::slice_data(
        projected_data, db::get<subcell::Tags::Mesh<Dim>>(*box).extents(),
        ghost_zone_size, directions_to_slice, rdmp_size,
        directions_not_to_slice);
    for (const auto& [directional_element_id, interpolator] :
         db::get<evolution::dg::subcell::Tags::InterpolatorsFromDgToNeighborFd<
             Dim>>(*box)) {
      // NOTE: no orienting is needed, that's done by the interpolation.
      ASSERT(interpolator.has_value(),
             "All interpolators must have values. The std::optional is used "
             "for some local optimizations.");
      DataVector& data_in_direction = all_neighbor_data_for_reconstruction->at(
          directional_element_id.direction());
      gsl::span result_in_direction{data_in_direction.data(),
                                    data_in_direction.size() - rdmp_size};
      interpolator->interpolate(
          make_not_null(&result_in_direction),
          {data_to_project.data(), data_to_project.size()});
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
