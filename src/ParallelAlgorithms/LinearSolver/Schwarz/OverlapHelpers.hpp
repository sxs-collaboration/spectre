// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <limits>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace LinearSolver::Schwarz {

/// Identifies a subdomain region that overlaps with another element
template <size_t Dim>
using OverlapId = std::pair<Direction<Dim>, ElementId<Dim>>;

/// Data structure that can store the `ValueType` on each possible overlap of an
/// element-centered subdomain with its neighbors. Overlaps are identified by
/// their `OverlapId`.
template <size_t Dim, typename ValueType>
using OverlapMap =
    FixedHashMap<maximum_number_of_neighbors(Dim), OverlapId<Dim>, ValueType,
                 boost::hash<OverlapId<Dim>>>;

/*!
 * \brief The number of points that an overlap extends into the `volume_extent`
 *
 * In a dimension where an element has `volume_extent` points, the overlap
 * extent is the largest number under these constraints:
 *
 * - It is at most `max_overlap`.
 * - It is smaller than the `volume_extent`.
 *
 * This means the overlap extent is always smaller than the `volume_extent`. The
 * reason for this constraint is that we define the _width_ of the overlap as
 * the element-logical coordinate distance from the face of the element to the
 * first collocation point _outside_ the overlap extent. Therefore, even an
 * overlap region that covers the full element in width does not include the
 * collocation point on the opposite side of the element.
 *
 * Here's a few notes on the definition of the overlap extent and width:
 *
 * - A typical smooth weighting function goes to zero at the overlap width, so
 *   if the grid points located at the overlap width were included in the
 *   subdomain, their solutions would not contribute to the weighted sum of
 *   subdomain solutions.
 * - Defining the overlap width as the distance to the first point _outside_ the
 *   overlap extent makes it non-zero even for a single point of overlap into a
 *   Gauss-Lobatto grid (which has points located at the element face).
 * - Boundary contributions for many (but not all) discontinuous Galerkin
 *   schemes on Gauss-Lobatto grids are limited to the grid points on the
 *   element face, e.g. for a DG operator that is pre-multiplied by the mass
 *   matrix, or one where boundary contributions are lifted using the diagonal
 *   mass-matrix approximation. Not including the grid points facing away from
 *   the subdomain in the overlap allows to ignore that face altogether in the
 *   subdomain operator.
 */
size_t overlap_extent(size_t volume_extent, size_t max_overlap) noexcept;

/*!
 * \brief Total number of grid points in an overlap region that extends
 * `overlap_extent` points into the `volume_extents` from either side in the
 * `overlap_dimension`
 *
 * The overlap region has `overlap_extent` points in the `overlap_dimension`,
 * and `volume_extents` points in the other dimensions. The number of grid
 * points returned by this function is the product of these extents.
 */
template <size_t Dim>
size_t overlap_num_points(const Index<Dim>& volume_extents,
                          size_t overlap_extent,
                          size_t overlap_dimension) noexcept;

/*!
 * \brief Width of an overlap extending `overlap_extent` points into the
 * `collocation_points` from either side.
 *
 * The "width" of an overlap is the element-logical coordinate distance from the
 * element boundary to the first collocation point outside the overlap region in
 * the overlap dimension, i.e. the dimension perpendicular to the element face.
 * See `LinearSolver::Schwarz::overlap_extent` for details.
 *
 * This function assumes the `collocation_points` are mirrored around 0.
 */
double overlap_width(size_t overlap_extent,
                     const DataVector& collocation_points) noexcept;

/*!
 * \brief Iterate over grid points in a region that extends partially into the
 * volume
 *
 * Here's an example how to use this iterator:
 *
 * \snippet Test_OverlapHelpers.cpp overlap_iterator
 */
class OverlapIterator {
 public:
  template <size_t Dim>
  OverlapIterator(const Index<Dim>& volume_extents, size_t overlap_extent,
                  const Direction<Dim>& direction) noexcept;

  explicit operator bool() const noexcept;

  OverlapIterator& operator++();

  /// Offset into a DataVector that holds full volume data
  size_t volume_offset() const noexcept;

  /// Offset into a DataVector that holds data only on the overlap region
  size_t overlap_offset() const noexcept;

  void reset() noexcept;

 private:
  size_t size_ = std::numeric_limits<size_t>::max();
  size_t num_slices_ = std::numeric_limits<size_t>::max();
  size_t stride_ = std::numeric_limits<size_t>::max();
  size_t stride_count_ = std::numeric_limits<size_t>::max();
  size_t jump_ = std::numeric_limits<size_t>::max();
  size_t initial_offset_ = std::numeric_limits<size_t>::max();
  size_t volume_offset_ = std::numeric_limits<size_t>::max();
  size_t overlap_offset_ = std::numeric_limits<size_t>::max();
};

/// @{
/// The part of the tensor data that lies within the overlap region
template <size_t Dim, typename DataType, typename... TensorStructure>
void data_on_overlap(const gsl::not_null<Tensor<DataType, TensorStructure...>*>
                         restricted_tensor,
                     const Tensor<DataType, TensorStructure...>& tensor,
                     const Index<Dim>& volume_extents,
                     const size_t overlap_extent,
                     const Direction<Dim>& direction) noexcept {
  for (OverlapIterator overlap_iterator{volume_extents, overlap_extent,
                                        direction};
       overlap_iterator; ++overlap_iterator) {
    for (size_t tensor_component = 0; tensor_component < tensor.size();
         ++tensor_component) {
      (*restricted_tensor)[tensor_component][overlap_iterator
                                                 .overlap_offset()] =
          tensor[tensor_component][overlap_iterator.volume_offset()];
    }
  }
}

template <size_t Dim, typename DataType, typename... TensorStructure>
Tensor<DataType, TensorStructure...> data_on_overlap(
    const Tensor<DataType, TensorStructure...>& tensor,
    const Index<Dim>& volume_extents, const size_t overlap_extent,
    const Direction<Dim>& direction) noexcept {
  Tensor<DataType, TensorStructure...> restricted_tensor{overlap_num_points(
      volume_extents, overlap_extent, direction.dimension())};
  data_on_overlap(make_not_null(&restricted_tensor), tensor, volume_extents,
                  overlap_extent, direction);
  return restricted_tensor;
}

namespace detail {
template <size_t Dim>
void data_on_overlap_impl(double* overlap_data, const double* volume_data,
                          size_t num_components,
                          const Index<Dim>& volume_extents,
                          size_t overlap_extent,
                          const Direction<Dim>& direction) noexcept;
}  // namespace detail

template <size_t Dim, typename OverlapTagsList, typename VolumeTagsList>
void data_on_overlap(
    const gsl::not_null<Variables<OverlapTagsList>*> overlap_data,
    const Variables<VolumeTagsList>& volume_data,
    const Index<Dim>& volume_extents, const size_t overlap_extent,
    const Direction<Dim>& direction) noexcept {
  constexpr size_t num_components =
      Variables<VolumeTagsList>::number_of_independent_components;
  ASSERT(volume_data.number_of_grid_points() == volume_extents.product(),
         "volume_data has wrong number of grid points.  Expected "
             << volume_extents.product() << ", got "
             << volume_data.number_of_grid_points());
  ASSERT(overlap_data->number_of_grid_points() ==
             overlap_num_points(volume_extents, overlap_extent,
                                direction.dimension()),
         "overlap_data has wrong number of grid points.  Expected "
             << overlap_num_points(volume_extents, overlap_extent,
                                   direction.dimension())
             << ", got " << overlap_data->number_of_grid_points());
  detail::data_on_overlap_impl(overlap_data->data(), volume_data.data(),
                               num_components, volume_extents, overlap_extent,
                               direction);
}

template <size_t Dim, typename TagsList>
Variables<TagsList> data_on_overlap(const Variables<TagsList>& volume_data,
                                    const Index<Dim>& volume_extents,
                                    const size_t overlap_extent,
                                    const Direction<Dim>& direction) noexcept {
  Variables<TagsList> overlap_data{overlap_num_points(
      volume_extents, overlap_extent, direction.dimension())};
  data_on_overlap(make_not_null(&overlap_data), volume_data, volume_extents,
                  overlap_extent, direction);
  return overlap_data;
}
/// @}

namespace detail {
template <size_t Dim>
void add_overlap_data_impl(double* volume_data, const double* overlap_data,
                           size_t num_components,
                           const Index<Dim>& volume_extents,
                           size_t overlap_extent,
                           const Direction<Dim>& direction) noexcept;
}  // namespace detail

/// Add the `overlap_data` to the `volume_data`
template <size_t Dim, typename VolumeTagsList, typename OverlapTagsList>
void add_overlap_data(
    const gsl::not_null<Variables<VolumeTagsList>*> volume_data,
    const Variables<OverlapTagsList>& overlap_data,
    const Index<Dim>& volume_extents, const size_t overlap_extent,
    const Direction<Dim>& direction) noexcept {
  constexpr size_t num_components =
      Variables<VolumeTagsList>::number_of_independent_components;
  ASSERT(volume_data->number_of_grid_points() == volume_extents.product(),
         "volume_data has wrong number of grid points.  Expected "
             << volume_extents.product() << ", got "
             << volume_data->number_of_grid_points());
  ASSERT(overlap_data.number_of_grid_points() ==
             overlap_num_points(volume_extents, overlap_extent,
                                direction.dimension()),
         "overlap_data has wrong number of grid points.  Expected "
             << overlap_num_points(volume_extents, overlap_extent,
                                   direction.dimension())
             << ", got " << overlap_data.number_of_grid_points());
  detail::add_overlap_data_impl(volume_data->data(), overlap_data.data(),
                                num_components, volume_extents, overlap_extent,
                                direction);
}

/// @{
/// Extend the overlap data to the full mesh by filling it with zeros outside
/// the overlap region
template <size_t Dim, typename ExtendedTagsList, typename OverlapTagsList>
void extended_overlap_data(
    const gsl::not_null<Variables<ExtendedTagsList>*> extended_data,
    const Variables<OverlapTagsList>& overlap_data,
    const Index<Dim>& volume_extents, const size_t overlap_extent,
    const Direction<Dim>& direction) noexcept {
  *extended_data = Variables<ExtendedTagsList>{volume_extents.product(), 0.};
  add_overlap_data(extended_data, overlap_data, volume_extents, overlap_extent,
                   direction);
}

template <size_t Dim, typename TagsList>
Variables<TagsList> extended_overlap_data(
    const Variables<TagsList>& overlap_data, const Index<Dim>& volume_extents,
    const size_t overlap_extent, const Direction<Dim>& direction) noexcept {
  Variables<TagsList> extended_data{volume_extents.product()};
  extended_overlap_data(make_not_null(&extended_data), overlap_data,
                        volume_extents, overlap_extent, direction);
  return extended_data;
}
/// @}

}  // namespace LinearSolver::Schwarz
