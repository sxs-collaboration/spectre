// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
/*!
 * \brief Project the DG package data to the subcells. Data received from a
 * neighboring element doing DG is always projected, while the data we sent to
 * our neighbors before doing a rollback from DG to subcell is only projected if
 * `OverwriteInternalMortarData` is `true`.
 *
 * In order for the hybrid DG-FD/FV scheme to be conservative between elements
 * using DG and elements using subcell, the boundary terms must be the same on
 * both elements. In practice this means the boundary corrections \f$G+D\f$ must
 * be computed on the same grid. Consider the element doing subcell which
 * receives data from an element doing DG. In this case the DG element's
 * ingredients going into \f$G+D\f$ are projected to the subcells and then
 * \f$G+D\f$ are computed on the subcells. Similarly, for strict conservation
 * the element doing DG must first project the data it sent to the neighbor to
 * the subcells, then compute \f$G+D\f$ on the subcells, and finally reconstrct
 * \f$G+D\f$ back to the DG grid before lifting \f$G+D\f$ to the volume.
 *
 * This function updates the `packaged_data` (ingredients into \f$G+D\f$)
 * received by an element doing subcell by projecting the neighbor's DG data
 * onto the subcells. Note that this is only half of what is required for strict
 * conservation, the DG element must also compute \f$G+D\f$ on the subcells.
 * Note that we currently do not perform the other half of the correction
 * needed to be strictly conservative.
 *
 * If we are retaking a time step after the DG step failed then maintaining
 * conservation requires additional care. If `OverwriteInternalMortarData` is
 * `true` then the local (the element switching from DG to subcell) ingredients
 * into \f$G+D\f$ are projected and overwrite the data computed from
 * the FD reconstruction to the interface. However, even this is insufficient to
 * guarantee conservation. To guarantee conservation (which we do not currently
 * do) the correction \f$G+D\f$ must be computed on the DG grid and then
 * projected to the subcells.
 *
 * Note that our practical experience shows that since the DG-subcell hybrid
 * scheme switches to the subcell solver _before_ the local solution contains
 * discontinuities, strict conservation is not necessary between DG and FD/FV
 * regions. This was also observed with a block-adaptive finite difference AMR
 * code \cite CHEN2016604
 */
template <bool OverwriteInternalMortarData, size_t Dim,
          typename DgPackageFieldTags>
void correct_package_data(
    const gsl::not_null<Variables<DgPackageFieldTags>*> lower_packaged_data,
    const gsl::not_null<Variables<DgPackageFieldTags>*> upper_packaged_data,
    const size_t logical_dimension_to_operate_in, const Element<Dim>& element,
    const Mesh<Dim>& subcell_volume_mesh,
    const std::unordered_map<
        std::pair<Direction<Dim>, ElementId<Dim>>,
        evolution::dg::MortarData<Dim>,
        boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
        mortar_data) noexcept {
  const Direction<Dim> upper_direction{logical_dimension_to_operate_in,
                                       Side::Upper};
  const Direction<Dim> lower_direction{logical_dimension_to_operate_in,
                                       Side::Lower};
  const bool has_upper_neighbor = element.neighbors().contains(upper_direction);
  const bool has_lower_neighbor = element.neighbors().contains(lower_direction);
  const std::pair upper_mortar_id =
      has_upper_neighbor
          ? std::pair{upper_direction,
                      *element.neighbors().at(upper_direction).begin()}
          : std::pair<Direction<Dim>, ElementId<Dim>>{};
  const std::pair lower_mortar_id =
      has_lower_neighbor
          ? std::pair{lower_direction,
                      *element.neighbors().at(lower_direction).begin()}
          : std::pair<Direction<Dim>, ElementId<Dim>>{};

  Index<Dim> subcell_extents_with_faces = subcell_volume_mesh.extents();
  ++subcell_extents_with_faces[logical_dimension_to_operate_in];
  const Mesh<Dim - 1>& subcell_face_mesh =
      subcell_volume_mesh.slice_away(logical_dimension_to_operate_in);

  const auto project_dg_data_to_subcells =
      [logical_dimension_to_operate_in, &subcell_extents_with_faces,
       &subcell_face_mesh](const gsl::not_null<Variables<DgPackageFieldTags>*>
                               subcell_packaged_data,
                           const size_t subcell_index,
                           const Mesh<Dim - 1>& neighbor_face_mesh,
                           const std::vector<double>& neighbor_data) noexcept {
        const double* slice_data = neighbor_data.data();
        // Warning: projected_data can't be inside the `if constexpr` since that
        // would lead to a dangling pointer.
        std::vector<double> projected_data{};
        if constexpr (Dim > 1) {
          projected_data.resize(
              Variables<DgPackageFieldTags>::number_of_independent_components *
              subcell_face_mesh.number_of_grid_points());
          evolution::dg::subcell::fd::detail::project_impl(
              gsl::make_span(projected_data.data(), projected_data.size()),
              gsl::make_span(neighbor_data.data(), neighbor_data.size()),
              neighbor_face_mesh, subcell_face_mesh.extents());
          slice_data = projected_data.data();
        } else {
          (void)subcell_face_mesh;
          (void)projected_data;
        }
        const size_t volume_grid_points = subcell_extents_with_faces.product();
        const size_t slice_grid_points =
            subcell_extents_with_faces
                .slice_away(logical_dimension_to_operate_in)
                .product();
        double* const volume_data = subcell_packaged_data->data();
        for (SliceIterator si(subcell_extents_with_faces,
                              logical_dimension_to_operate_in, subcell_index);
             si; ++si) {
          for (size_t i = 0;
               i <
               Variables<DgPackageFieldTags>::number_of_independent_components;
               ++i) {
            // clang-tidy: do not use pointer arithmetic
            volume_data[si.volume_offset() +
                        i * volume_grid_points] =  // NOLINT
                slice_data[si.slice_offset() +
                           i * slice_grid_points];  // NOLINT
          }
        }
      };

  // Project DG data to the subcells
  if (auto neighbor_mortar_data_it = mortar_data.find(upper_mortar_id);
      has_upper_neighbor and neighbor_mortar_data_it != mortar_data.end()) {
    if (neighbor_mortar_data_it->second.neighbor_mortar_data().has_value()) {
      project_dg_data_to_subcells(
          upper_packaged_data,
          subcell_extents_with_faces[logical_dimension_to_operate_in] - 1,
          neighbor_mortar_data_it->second.neighbor_mortar_data()->first,
          neighbor_mortar_data_it->second.neighbor_mortar_data()->second);
    }
    if constexpr (OverwriteInternalMortarData) {
      if (neighbor_mortar_data_it->second.local_mortar_data().has_value()) {
        project_dg_data_to_subcells(
            lower_packaged_data,
            subcell_extents_with_faces[logical_dimension_to_operate_in] - 1,
            neighbor_mortar_data_it->second.local_mortar_data()->first,
            neighbor_mortar_data_it->second.local_mortar_data()->second);
      }
    }
  }
  if (auto neighbor_mortar_data_it = mortar_data.find(lower_mortar_id);
      has_lower_neighbor and neighbor_mortar_data_it != mortar_data.end()) {
    if (neighbor_mortar_data_it->second.neighbor_mortar_data().has_value()) {
      project_dg_data_to_subcells(
          lower_packaged_data, 0,
          neighbor_mortar_data_it->second.neighbor_mortar_data()->first,
          neighbor_mortar_data_it->second.neighbor_mortar_data()->second);
    }
    if constexpr (OverwriteInternalMortarData) {
      if (neighbor_mortar_data_it->second.local_mortar_data().has_value()) {
        project_dg_data_to_subcells(
            upper_packaged_data, 0,
            neighbor_mortar_data_it->second.local_mortar_data()->first,
            neighbor_mortar_data_it->second.local_mortar_data()->second);
      }
    }
  }
}
}  // namespace evolution::dg::subcell
