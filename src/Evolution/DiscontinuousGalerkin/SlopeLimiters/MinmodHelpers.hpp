// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tags.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
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

// Allocate the buffers `u_lin_buffer` and `boundary_buffer` to the correct
// sizes expected by `troubled_cell_indicator` for its arguments
template <size_t VolumeDim>
void allocate_buffers(
    gsl::not_null<std::unique_ptr<double[], decltype(&free)>*>
        contiguous_buffer,
    gsl::not_null<DataVector*> u_lin_buffer,
    gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const Mesh<VolumeDim>& mesh) noexcept;

// In each direction, average the size of all different neighbors in that
// direction. Note that only the component of neighor_size that is normal
// to the face is needed (and, therefore, computed).
//
// Expects type `PackagedData` to contain a variable `element_size` that is a
// `std::array<double, VolumeDim>`.
template <size_t VolumeDim, typename PackagedData>
DirectionMap<VolumeDim, double> compute_effective_neighbor_sizes(
    const Element<VolumeDim>& element,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  DirectionMap<VolumeDim, double> result;
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const auto& externals = element.external_boundaries();
    const bool neighbors_in_this_dir = (externals.find(dir) == externals.end());
    if (neighbors_in_this_dir) {
      const double effective_neighbor_size =
          [&dir, &element, &neighbor_data ]() noexcept {
        const size_t dim = dir.dimension();
        const auto& neighbor_ids = element.neighbors().at(dir).ids();
        double size_accumulate = 0.;
        for (const auto& id : neighbor_ids) {
          size_accumulate += gsl::at(
              neighbor_data.at(std::make_pair(dir, id)).element_size, dim);
        }
        return size_accumulate / neighbor_ids.size();
      }
      ();
      result.insert(std::make_pair(dir, effective_neighbor_size));
    }
  }
  return result;
}

// In each direction, average the mean of the specified tensor component over
// all different neighbors that direction. This produces one effective neighbor
// per direction.
//
// Expects type `PackagedData` to contain a variable `means` that is a
// `TaggedTuple<Tags::Mean<Tags>...>`. Tags... must contain Tag, the tag
// specifying the tensor to work with.
template <typename Tag, size_t VolumeDim, typename PackagedData>
DirectionMap<VolumeDim, double> compute_effective_neighbor_means(
    const Element<VolumeDim>& element, const size_t tensor_storage_index,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  DirectionMap<VolumeDim, double> result;
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const auto& externals = element.external_boundaries();
    const bool neighbors_in_this_dir = (externals.find(dir) == externals.end());
    if (neighbors_in_this_dir) {
      const double effective_neighbor_mean =
          [&dir, &element, &neighbor_data, &tensor_storage_index ]() noexcept {
        const auto& neighbor_ids = element.neighbors().at(dir).ids();
        double mean_accumulate = 0.0;
        for (const auto& id : neighbor_ids) {
          mean_accumulate += tuples::get<::Tags::Mean<Tag>>(
              neighbor_data.at(std::make_pair(dir, id))
                  .means)[tensor_storage_index];
        }
        return mean_accumulate / neighbor_ids.size();
      }
      ();
      result.insert(std::make_pair(dir, effective_neighbor_mean));
    }
  }
  return result;
}

}  // namespace Minmod_detail
}  // namespace SlopeLimiters
