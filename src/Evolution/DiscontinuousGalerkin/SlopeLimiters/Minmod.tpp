// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <pup.h>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

namespace SlopeLimiters {
namespace Minmod_detail {

// Implements the minmod limiter for one Tensor<DataVector> at a time.
template <size_t VolumeDim, typename Tag, typename PackagedData>
bool limit_one_tensor(
    const gsl::not_null<db::item_type<Tag>*> tensor,
    const gsl::not_null<DataVector*> u_lin_buffer,
    const gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const SlopeLimiters::MinmodType& minmod_type, const double tvbm_constant,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                               gsl::span<std::pair<size_t, size_t>>>,
                     VolumeDim>& volume_and_slice_indices) noexcept {
  // True if the mesh is linear-order in every direction
  const bool mesh_is_linear = (mesh.extents() == Index<VolumeDim>(2));
  const bool minmod_type_is_linear =
      (minmod_type != SlopeLimiters::MinmodType::LambdaPiN);
  const bool using_linear_limiter_on_non_linear_mesh =
      minmod_type_is_linear and not mesh_is_linear;

  // In each direction, average the size of all different neighbors in that
  // direction. Note that only the component of neighor_size that is normal
  // to the face is needed (and, therefore, computed). Note that this average
  // does not depend on the solution on the neighboring elements, so could be
  // precomputed outside of `limit_one_tensor`. Changing the code to
  // precompute the average may or may not be a measurable optimization.
  const auto effective_neighbor_sizes =
      compute_effective_neighbor_sizes(element, neighbor_data);

  bool some_component_was_limited = false;
  for (size_t i = 0; i < tensor->size(); ++i) {
    // In each direction, average the mean of the i'th tensor component over
    // all different neighbors in that direction. This produces one effective
    // neighbor per direction.
    const auto effective_neighbor_means =
        compute_effective_neighbor_means<Tag>(element, i, neighbor_data);

    DataVector& u = (*tensor)[i];
    double u_mean;
    std::array<double, VolumeDim> u_limited_slopes{};
    const bool reduce_slope = troubled_cell_indicator(
        make_not_null(&u_mean), make_not_null(&u_limited_slopes), u_lin_buffer,
        boundary_buffer, minmod_type, tvbm_constant, u, element, mesh,
        element_size, effective_neighbor_means, effective_neighbor_sizes,
        volume_and_slice_indices);

    if (reduce_slope or using_linear_limiter_on_non_linear_mesh) {
      u = u_mean;
      for (size_t d = 0; d < VolumeDim; ++d) {
        u += logical_coords.get(d) * gsl::at(u_limited_slopes, d);
      }
      some_component_was_limited = true;
    }
  }  // end for loop over tensor components

  return some_component_was_limited;
}

}  // namespace Minmod_detail

template <size_t VolumeDim, typename... Tags>
Minmod<VolumeDim, tmpl::list<Tags...>>::Minmod(
    const MinmodType minmod_type, const double tvbm_constant,
    const bool disable_for_debugging) noexcept
    : minmod_type_(minmod_type),
      tvbm_constant_(tvbm_constant),
      disable_for_debugging_(disable_for_debugging) {
  ASSERT(tvbm_constant >= 0.0, "The TVBM constant must be non-negative.");
}

template <size_t VolumeDim, typename... Tags>
void Minmod<VolumeDim, tmpl::list<Tags...>>::pup(PUP::er& p) noexcept {
  p | minmod_type_;
  p | tvbm_constant_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim, typename... Tags>
void Minmod<VolumeDim, tmpl::list<Tags...>>::package_data(
    const gsl::not_null<PackagedData*>& packaged_data,
    const db::item_type<Tags>&... tensors, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not initialize packaged_data
    return;
  }

  const auto wrap_compute_means =
      [&mesh, &packaged_data ](auto tag, const auto& tensor) noexcept {
    for (size_t i = 0; i < tensor.size(); ++i) {
      // Compute the mean using the local orientation of the tensor and mesh:
      // this avoids the work of reorienting the tensor while giving the same
      // result.
      get<::Tags::Mean<decltype(tag)>>(packaged_data->means)[i] =
          mean_value(tensor[i], mesh);
    }
    return '0';
  };
  expand_pack(wrap_compute_means(Tags{}, tensors)...);
  packaged_data->element_size =
      orientation_map.permute_from_neighbor(element_size);
}

template <size_t VolumeDim, typename... Tags>
bool Minmod<VolumeDim, tmpl::list<Tags...>>::operator()(
    const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Optimization: allocate temporary buffer to be used in `limit_one_tensor`
  std::unique_ptr<double[], decltype(&free)> contiguous_buffer(nullptr, &free);
  DataVector u_lin_buffer{};
  std::array<DataVector, VolumeDim> boundary_buffer{};
  Minmod_detail::allocate_buffers(make_not_null(&contiguous_buffer),
                                  make_not_null(&u_lin_buffer),
                                  make_not_null(&boundary_buffer), mesh);

  // Optimization: precompute the slice indices since this is (surprisingly)
  // expensive
  const auto volume_and_slice_buffer_and_indices =
      volume_and_slice_indices(mesh.extents());
  const auto& volume_and_slice_indices =
      volume_and_slice_buffer_and_indices.second;

  bool limiter_activated = false;
  const auto wrap_limit_one_tensor = [
    this, &limiter_activated, &element, &mesh, &logical_coords, &element_size,
    &neighbor_data, &u_lin_buffer, &boundary_buffer, &volume_and_slice_indices
  ](auto tag, const auto& tensor) noexcept {
    limiter_activated =
        Minmod_detail::limit_one_tensor<VolumeDim, decltype(tag)>(
            tensor, &u_lin_buffer, &boundary_buffer, minmod_type_,
            tvbm_constant_, element, mesh, logical_coords, element_size,
            neighbor_data, volume_and_slice_indices) or
        limiter_activated;
    return '0';
  };
  expand_pack(wrap_limit_one_tensor(Tags{}, tensors)...);
  return limiter_activated;
}

template <size_t LocalDim, typename LocalTagList>
bool operator==(const Minmod<LocalDim, LocalTagList>& lhs,
                const Minmod<LocalDim, LocalTagList>& rhs) noexcept {
  return lhs.minmod_type_ == rhs.minmod_type_ and
         lhs.tvbm_constant_ == rhs.tvbm_constant_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim, typename TagList>
bool operator!=(const Minmod<VolumeDim, TagList>& lhs,
                const Minmod<VolumeDim, TagList>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace SlopeLimiters
