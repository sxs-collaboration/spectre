// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"

#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace LinearSolver::Schwarz {

size_t overlap_extent(const size_t volume_extent,
                      const size_t max_overlap) noexcept {
  return std::min(max_overlap, volume_extent - 1);
}

namespace {
template <size_t Dim>
void assert_overlap_extent(const Index<Dim>& volume_extents,
                           const size_t overlap_extent,
                           const size_t overlap_dimension) noexcept {
  ASSERT(overlap_dimension < Dim,
         "Invalid dimension '" << overlap_dimension << "' in " << Dim << "D.");
  ASSERT(overlap_extent <= volume_extents[overlap_dimension],
         "Overlap extent '" << overlap_extent << "' exceeds volume extents '"
                            << volume_extents << "' in overlap dimension '"
                            << overlap_dimension << "'.");
}
}  // namespace

template <size_t Dim>
size_t overlap_num_points(const Index<Dim>& volume_extents,
                          const size_t overlap_extent,
                          const size_t overlap_dimension) noexcept {
  assert_overlap_extent(volume_extents, overlap_extent, overlap_dimension);
  return volume_extents.slice_away(overlap_dimension).product() *
         overlap_extent;
}

double overlap_width(const size_t overlap_extent,
                     const DataVector& collocation_points) noexcept {
  ASSERT(overlap_extent < collocation_points.size(),
         "Overlap extent is "
             << overlap_extent
             << " but must be strictly smaller than the number of grid points ("
             << collocation_points.size() << ") to compute an overlap width.");
  for (size_t i = 0; i < collocation_points.size() / 2; ++i) {
    ASSERT(equal_within_roundoff(
               -collocation_points[i],
               collocation_points[collocation_points.size() - i - 1]),
           "Assuming the 'collocation_points' are symmetric, but they are not: "
               << collocation_points);
  }
  // The overlap boundary index lies one point outside the region covered by the
  // overlap coordinates (see `LinearSolver::Schwarz::overlap_extent`).
  return collocation_points[overlap_extent] - collocation_points[0];
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data) \
  template size_t overlap_num_points(const Index<DIM(data)>&, size_t, size_t);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace LinearSolver::Schwarz
