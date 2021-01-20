// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Actions::detail {
// Helper function to get parameter packs so we can forward `Tensor`s instead
// of `Variables` to the boundary corrections. Returns the maximum absolute
// char speed on the face, which can be used for setting or monitoring the CFL
// without having to compute the speeds for each dimension in the volume.
// Whether using only the face speeds is accurate enough to guarantee
// stability is yet to be determined. However, if the CFL condition is
// violated on the boundaries we are definitely in trouble, so it can at least
// be used as a cheap diagnostic.
template <typename System, typename BoundaryCorrection,
          typename... PackagedFieldTags, typename... ProjectedFieldTags,
          typename... ProjectedFieldTagsForCorrection, size_t Dim,
          typename... VolumeArgs>
double dg_package_data(
    const gsl::not_null<Variables<tmpl::list<PackagedFieldTags...>>*>
        packaged_data,
    const BoundaryCorrection& boundary_correction,
    const Variables<tmpl::list<ProjectedFieldTags...>>& projected_fields,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    tmpl::list<ProjectedFieldTagsForCorrection...> /*meta*/,
    const VolumeArgs&... volume_args) noexcept {
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};
  if (mesh_velocity.has_value()) {
    normal_dot_mesh_velocity =
        dot_product(*mesh_velocity, unit_normal_covector);
  }

  if constexpr (evolution::dg::Actions::detail::
                    has_inverse_spatial_metric_tag_v<System>) {
    return boundary_correction.dg_package_data(
        make_not_null(&get<PackagedFieldTags>(*packaged_data))...,
        get<ProjectedFieldTagsForCorrection>(projected_fields)...,
        unit_normal_covector,
        get<evolution::dg::Actions::detail::NormalVector<Dim>>(
            projected_fields),
        mesh_velocity, normal_dot_mesh_velocity, volume_args...);
  } else {
    return boundary_correction.dg_package_data(
        make_not_null(&get<PackagedFieldTags>(*packaged_data))...,
        get<ProjectedFieldTagsForCorrection>(projected_fields)...,
        unit_normal_covector, mesh_velocity, normal_dot_mesh_velocity,
        volume_args...);
  }
}

template <typename System, typename BoundaryCorrection,
          typename... PackagedFieldTags, typename... ProjectedFieldTags,
          typename... ProjectedFieldTagsForCorrection, size_t Dim,
          typename DbTagsList, typename... VolumeTags>
double dg_package_data(
    const gsl::not_null<Variables<tmpl::list<PackagedFieldTags...>>*>
        packaged_data,
    const BoundaryCorrection& boundary_correction,
    const Variables<tmpl::list<ProjectedFieldTags...>>& projected_fields,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    const db::DataBox<DbTagsList>& box, tmpl::list<VolumeTags...> /*meta*/,
    tmpl::list<ProjectedFieldTagsForCorrection...> /*meta*/) noexcept {
  return dg_package_data<System>(
      packaged_data, boundary_correction, projected_fields,
      unit_normal_covector, mesh_velocity,
      tmpl::list<ProjectedFieldTagsForCorrection...>{},
      db::get<VolumeTags>(box)...);
}
}  // namespace evolution::dg::Actions::detail
