// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/MinimumGridSpacing.hpp"

// For std::min
#include <algorithm>  // IWYU pragma: keep
#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {
template <size_t Dim, typename Frame>
std::array<double, Dim> nth_point(const tnsr::I<DataVector, Dim, Frame>& tensor,
                                  const size_t index) noexcept {
  std::array<double, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = tensor.get(d)[index];
  }
  return result;
}
}  // namespace

template <size_t Dim, typename Frame>
double minimum_grid_spacing(
    const Index<Dim>& extents,
    const tnsr::I<DataVector, Dim, Frame>& coords) noexcept {
  double minimum_spacing = std::numeric_limits<double>::max();
  for (IndexIterator<Dim> point(extents); point; ++point) {
    const auto point_coords = nth_point(coords, point.collapsed_index());
    // We assume that the coordinates are not too distorted in that
    // the closest point to a given point cannot be off by more than
    // one index in any dimension.
    for (IndexIterator<Dim> offset(Index<Dim>(3)); offset; ++offset) {
      if (*offset == Index<Dim>(1)) {
        // All the remaining directions are the opposite of ones we
        // have already checked.  They will be checked from the other
        // point.
        break;
      }
      // On initialization this is one-indexed.
      Index<Dim> other_point(point->indices() + offset->indices());
      for (size_t d = 0; d < Dim; ++d) {
        if (other_point[d] == 0 or other_point[d] == extents[d] + 1) {
          goto next_offset;
        }
        --other_point[d];
      }
      minimum_spacing = std::min(
          minimum_spacing,
          magnitude(nth_point(coords, collapsed_index(other_point, extents)) -
                    point_coords));

    next_offset:;
    }
  }

  return minimum_spacing;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)            \
  template double minimum_grid_spacing( \
      const Index<DIM(data)>& extents,  \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& coords) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
