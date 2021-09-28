// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <iterator>
#include <limits>
#include <memory>
#include <pup.h>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

namespace Limiters::Minmod_detail {

// This function combines the evaluation of the troubled-cell indicator with the
// computation of the post-limiter reduced slopes. The returned bool indicates
// whether the slopes are to be reduced. The slopes themselves are returned by
// pointer.
//
// Note: This function is only made available in this header file to facilitate
// testing.
template <size_t VolumeDim>
bool minmod_limited_slopes(
    gsl::not_null<DataVector*> u_lin_buffer,
    gsl::not_null<BufferWrapper<VolumeDim>*> buffer,
    gsl::not_null<double*> u_mean,
    gsl::not_null<std::array<double, VolumeDim>*> u_limited_slopes,
    Limiters::MinmodType minmod_type, double tvb_constant, const DataVector& u,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes) noexcept;

// Implements the minmod limiter for one Tensor<DataVector> at a time.
template <size_t VolumeDim, typename Tag, typename PackagedData>
bool minmod_impl(
    const gsl::not_null<DataVector*> u_lin_buffer,
    const gsl::not_null<BufferWrapper<VolumeDim>*> buffer,
    const gsl::not_null<typename Tag::type*> tensor,
    const Limiters::MinmodType minmod_type, const double tvb_constant,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const tnsr::I<DataVector, VolumeDim, Frame::ElementLogical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  // True if the mesh is linear-order in every direction
  const bool mesh_is_linear = (mesh.extents() == Index<VolumeDim>(2));
  const bool minmod_type_is_linear =
      (minmod_type != Limiters::MinmodType::LambdaPiN);
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
        compute_effective_neighbor_means<Tag>(i, element, neighbor_data);

    DataVector& u = (*tensor)[i];
    double u_mean = std::numeric_limits<double>::signaling_NaN();
    std::array<double, VolumeDim> u_limited_slopes{};
    const bool reduce_slopes = minmod_limited_slopes(
        u_lin_buffer, buffer, make_not_null(&u_mean),
        make_not_null(&u_limited_slopes), minmod_type, tvb_constant, u, mesh,
        element, element_size, effective_neighbor_means,
        effective_neighbor_sizes);

    if (reduce_slopes or using_linear_limiter_on_non_linear_mesh) {
      u = u_mean;
      for (size_t d = 0; d < VolumeDim; ++d) {
        u += logical_coords.get(d) * gsl::at(u_limited_slopes, d);
      }
      some_component_was_limited = true;
    }
  }  // end for loop over tensor components

  return some_component_was_limited;
}

}  // namespace Limiters::Minmod_detail
