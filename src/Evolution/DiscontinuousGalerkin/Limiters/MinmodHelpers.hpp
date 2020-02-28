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
enum class Side;

namespace boost {
template <class T>
struct hash;
}  // namespace boost
/// \endcond

namespace Limiters {
namespace Minmod_detail {

// Encodes the return status of the tvb_corrected_minmod function.
struct MinmodResult {
  const double value;
  const bool activated;
};

// The TVB-corrected minmod function, see e.g. Cockburn reference Eq. 2.26.
MinmodResult tvb_corrected_minmod(double a, double b, double c,
                                  double tvb_scale) noexcept;

// Holds various optimization-related allocations for the Minmod TCI.
// There is no pup::er, because these allocations should be short-lived (i.e.,
// scoped within a single limiter invocation).
template <size_t VolumeDim>
class BufferWrapper {
 public:
  BufferWrapper() = delete;
  explicit BufferWrapper(const Mesh<VolumeDim>& mesh) noexcept;

 private:
  std::unique_ptr<double[], decltype(&free)> contiguous_boundary_buffer_;
  const std::pair<std::unique_ptr<std::pair<size_t, size_t>[], decltype(&free)>,
                  std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                                       gsl::span<std::pair<size_t, size_t>>>,
                             VolumeDim>>
      volume_and_slice_buffer_and_indices_ {};

 public:
  std::array<DataVector, VolumeDim> boundary_buffers{};
  const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                             gsl::span<std::pair<size_t, size_t>>>,
                   VolumeDim>& volume_and_slice_indices{};
};

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

// Compute an effective element-center-to-neighbor-center distance that accounts
// for the possibility of different refinement levels or discontinuous maps
// (e.g., at Block boundaries). Treated naively, these domain features can make
// a smooth solution appear to be non-smooth in the logical coordinates, which
// could potentially lead to the limiter triggering erroneously. This effective
// distance is used to scale the difference in the means, so that a linear
// function at a refinement or Block boundary will still appear smooth to the
// limiter. The factor is normalized to be 1.0 on a uniform grid.
//
// Note that this is not "by the book" Minmod, but an attempt to
// generalize Minmod to work on non-uniform grids.
template <size_t VolumeDim>
double effective_difference_to_neighbor(
    double u_mean, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes, size_t dim,
    const Side& side) noexcept;

}  // namespace Minmod_detail
}  // namespace Limiters
