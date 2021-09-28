// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t Dim>
struct Index;
template <size_t Dim>
struct Mesh;
template <size_t Dim>
struct OrientationMap;
/// \endcond

namespace dg {

/*!
 * \brief Distinguishes between field data, which can be projected to a mortar,
 * and extra data, which will not be projected.
 */
template <typename FieldTags, typename ExtraDataTags = tmpl::list<>>
struct SimpleBoundaryData {
  using field_tags = FieldTags;
  using extra_data_tags = ExtraDataTags;

  /// Data projected to the mortar mesh
  Variables<FieldTags> field_data;

  /// Data on the element face that needs no projection to the mortar mesh.
  /// This is a `TaggedTuple` to support non-tensor quantities. It also helps
  /// supporting an empty list of `ExtraDataTags`.
  tuples::tagged_tuple_from_typelist<ExtraDataTags> extra_data;

  SimpleBoundaryData() = default;
  SimpleBoundaryData(const SimpleBoundaryData&) = default;
  SimpleBoundaryData& operator=(const SimpleBoundaryData&) = default;
  SimpleBoundaryData(SimpleBoundaryData&&) = default;
  SimpleBoundaryData& operator=(SimpleBoundaryData&&) = default;
  ~SimpleBoundaryData() = default;

  explicit SimpleBoundaryData(const size_t num_points)
      : field_data{num_points}, extra_data{} {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | field_data;
    p | extra_data;
  }

  /// Project the `field_data` to the mortar
  ///
  /// \see `dg::project_to_mortar`
  template <size_t MortarDim>
  SimpleBoundaryData<FieldTags, ExtraDataTags> project_to_mortar(
      const Mesh<MortarDim>& face_mesh, const Mesh<MortarDim>& mortar_mesh,
      const MortarSize<MortarDim>& mortar_size) const {
    SimpleBoundaryData<FieldTags, ExtraDataTags> projected_data{};
    projected_data.field_data = dg::project_to_mortar(
        this->field_data, face_mesh, mortar_mesh, mortar_size);
    projected_data.extra_data = this->extra_data;
    return projected_data;
  }

  /// Orient the `field_data`
  ///
  /// \see `orient_variables_on_slice`
  template <size_t MortarDim>
  void orient_on_slice(
      const Index<MortarDim>& slice_extents, const size_t sliced_dim,
      const OrientationMap<MortarDim + 1>& orientation_of_neighbor) {
    this->field_data = orient_variables_on_slice(
        this->field_data, slice_extents, sliced_dim, orientation_of_neighbor);
  }
};

}  // namespace dg
