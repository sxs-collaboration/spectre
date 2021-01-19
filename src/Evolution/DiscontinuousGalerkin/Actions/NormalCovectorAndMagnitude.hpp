// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Actions::detail {
struct OneOverNormalVectorMagnitude : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct NormalVector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

/*!
 * \brief Computes the normal covector, magnitude of the unnormalized normal
 * covector, and in curved spacetimes the normal vector.
 *
 * The `fields_on_face` argument is used as a temporary buffer via the
 * `OneOverNormalVectorMagnitude` tag and also used to return the normalized
 * normal vector via the `NormalVector<Dim>` tag. The `fields_on_face` argument
 * also serves as the input for the inverse spatial metric when the spacetime is
 * curved. The `fields_on_face` argument must have tags `NormalVector<Dim>` and
 * `OneOverNormalVectorMagnitude`. If the system specifies
 * `System::inverse_spatial_metric_tag` then this tag must also be in the
 * Variables.
 */
template <typename System, size_t Dim, typename FieldsOnFaceTags>
void unit_normal_vector_and_covector_and_magnitude_impl(
    const gsl::not_null<Scalar<DataVector>*> face_normal_magnitude,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        unit_normal_covector,
    const gsl::not_null<Variables<FieldsOnFaceTags>*> fields_on_face,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unnormalized_normal_covector) noexcept {
  if constexpr (has_inverse_spatial_metric_tag_v<System>) {
    using inverse_spatial_metric_tag =
        typename System::inverse_spatial_metric_tag;
    auto& normal_vector = get<NormalVector<Dim>>(*fields_on_face);
    const auto& inverse_spatial_metric =
        get<inverse_spatial_metric_tag>(*fields_on_face);
    // Compute unnormalized normal vector
    for (size_t i = 0; i < Dim; ++i) {
      normal_vector.get(i) = inverse_spatial_metric.get(i, 0) *
                             get<0>(unnormalized_normal_covector);
      for (size_t j = 1; j < Dim; ++j) {
        normal_vector.get(i) += inverse_spatial_metric.get(i, j) *
                                unnormalized_normal_covector.get(j);
      }
    }
    // Perform normalization
    dot_product(face_normal_magnitude, normal_vector,
                unnormalized_normal_covector);
    get(*face_normal_magnitude) = sqrt(get(*face_normal_magnitude));
    auto& one_over_normal_vector_magnitude =
        get<OneOverNormalVectorMagnitude>(*fields_on_face);
    get(one_over_normal_vector_magnitude) = 1.0 / get(*face_normal_magnitude);
    for (size_t i = 0; i < Dim; ++i) {
      unit_normal_covector->get(i) = unnormalized_normal_covector.get(i) *
                                     get(one_over_normal_vector_magnitude);
      normal_vector.get(i) *= get(one_over_normal_vector_magnitude);
    }
  } else {
    magnitude(face_normal_magnitude, unnormalized_normal_covector);
    auto& one_over_normal_vector_magnitude =
        get<OneOverNormalVectorMagnitude>(*fields_on_face);
    get(one_over_normal_vector_magnitude) = 1.0 / get(*face_normal_magnitude);
    for (size_t i = 0; i < Dim; ++i) {
      unit_normal_covector->get(i) = unnormalized_normal_covector.get(i) *
                                     get(one_over_normal_vector_magnitude);
    }
  }
}

/*!
 * \brief Computes the normal vector, covector, and their magnitude on the face
 * in the given direction.
 *
 * The normal covector and its magnitude are returned using the
 * `normal_covector_quantities` function argument. The `fields_on_face` argument
 * is used as a temporary buffer via the `OneOverNormalVectorMagnitude` tag and
 * also used to return the normalized normal vector via the `NormalVector<Dim>`
 * tag. The `fields_on_face` argument also serves as the input for the inverse
 * spatial metric when the spacetime is curved. The `fields_on_face` argument
 * must have tags `NormalVector<Dim>` and `OneOverNormalVectorMagnitude`. If the
 * system specifies `System::inverse_spatial_metric_tag` then this tag must also
 * be in the Variables.
 */
template <typename System, size_t Dim, typename FieldsOnFaceTags>
void unit_normal_vector_and_covector_and_magnitude(
    const gsl::not_null<
        DirectionMap<Dim, std::optional<Variables<tmpl::list<
                              evolution::dg::Tags::MagnitudeOfNormal,
                              evolution::dg::Tags::NormalCovector<Dim>>>>>*>
        normal_covector_quantities,
    const gsl::not_null<Variables<FieldsOnFaceTags>*> fields_on_face,
    const Direction<Dim>& direction,
    const std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>&
        unnormalized_normal_covectors,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
        moving_mesh_map) noexcept {
  const bool mesh_is_moving = not moving_mesh_map.is_identity();
  const auto& unnormalized_normal_covector =
      unnormalized_normal_covectors.at(direction);

  if (auto& normal_covector_quantity =
          normal_covector_quantities->at(direction);
      detail::has_inverse_spatial_metric_tag_v<System> or mesh_is_moving or
      not normal_covector_quantity.has_value()) {
    if (not normal_covector_quantity.has_value()) {
      normal_covector_quantity =
          Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                               evolution::dg::Tags::NormalCovector<Dim>>>{
              fields_on_face->number_of_grid_points()};
    }
    detail::unit_normal_vector_and_covector_and_magnitude_impl<System>(
        make_not_null(&get<evolution::dg::Tags::MagnitudeOfNormal>(
            *normal_covector_quantity)),
        make_not_null(&get<evolution::dg::Tags::NormalCovector<Dim>>(
            *normal_covector_quantity)),
        fields_on_face, unnormalized_normal_covector);
  }
}
}  // namespace evolution::dg::Actions::detail
