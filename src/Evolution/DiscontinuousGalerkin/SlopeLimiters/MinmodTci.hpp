// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t Dim, typename T>
class DirectionMap;
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class Mesh;

namespace boost {
template <class T>
struct hash;
}  // namespace boost
/// \endcond

namespace SlopeLimiters {
namespace Minmod_detail {

// Implements the troubled-cell indicator corresponding to the Minmod limiter.
//
// The troubled-cell indicator (TCI) determines whether or not limiting is
// needed. See SlopeLimiters::Minmod for a full description of the Minmod
// limiter. Note that as an optimization, this TCI returns (by reference) some
// additional data that are used by the Minmod limiter in the case where the
// TCI returns true (i.e., the case where limiting is needed).
template <size_t VolumeDim>
bool troubled_cell_indicator(
    gsl::not_null<double*> u_mean,
    gsl::not_null<std::array<double, VolumeDim>*> u_limited_slopes,
    gsl::not_null<DataVector*> u_lin_buffer,
    gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const SlopeLimiters::MinmodType& minmod_type, double tvbm_constant,
    const DataVector& u, const Element<VolumeDim>& element,
    const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes,
    const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                               gsl::span<std::pair<size_t, size_t>>>,
                     VolumeDim>& volume_and_slice_indices) noexcept;

// Implements the minmod troubled-cell indicator on several tensors.
//
// Internally loops over all tensors and tensor components, calling the above
// function for each case. Returns true if any of the components needs limiting.
// Optimization: stops as soon as one component needs limiting, because no
// intermediate results are passed "up" out of this function.
//
// Expects type `PackagedData` to contain, as in SlopeLimiters::Minmod:
// - a variable `means` that is a `TaggedTuple<Tags::Mean<Tags>...>`
// - a variable `element_size` that is a `std::array<double, VolumeDim>`
template <size_t VolumeDim, typename PackagedData, typename... Tags>
bool troubled_cell_indicator(
    const db::item_type<Tags>&... tensors,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const SlopeLimiters::MinmodType& minmod_type, double tvbm_constant,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size) noexcept {
  // Optimization: allocate temporary buffer to be used in TCI
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

  const auto effective_neighbor_sizes =
      compute_effective_neighbor_sizes(element, neighbor_data);

  // Ideally, as soon as one component is found that needs limiting, then we
  // would exit the TCI early with return value `true`. But there is no natural
  // way to return early from the pack expansion. So we use this bool to keep
  // track of whether a previous component needed limiting... and if so, then
  // we simply skip any work.
  bool some_component_needs_limiting = false;
  const auto wrap_tci_one_tensor = [&](auto tag, const auto& tensor) noexcept {
    if (some_component_needs_limiting) {
      // Skip this tensor completely
      return '0';
    }

    for (size_t tensor_storage_index = 0; tensor_storage_index < tensor.size();
         ++tensor_storage_index) {
      const auto effective_neighbor_means =
          compute_effective_neighbor_means<decltype(tag)>(
              element, tensor_storage_index, neighbor_data);

      const DataVector& u = tensor[tensor_storage_index];
      double u_mean;
      std::array<double, VolumeDim> u_limited_slopes{};
      const bool reduce_slope = troubled_cell_indicator(
          make_not_null(&u_mean), make_not_null(&u_limited_slopes),
          make_not_null(&u_lin_buffer), make_not_null(&boundary_buffer),
          minmod_type, tvbm_constant, u, element, mesh, element_size,
          effective_neighbor_means, effective_neighbor_sizes,
          volume_and_slice_indices);

      if (reduce_slope) {
        some_component_needs_limiting = true;
        // Skip remaining components of this tensor
        return '0';
      }
    }
    return '0';
  };
  expand_pack(wrap_tci_one_tensor(Tags{}, tensors)...);
  return some_component_needs_limiting;
}

}  // namespace Minmod_detail
}  // namespace SlopeLimiters
