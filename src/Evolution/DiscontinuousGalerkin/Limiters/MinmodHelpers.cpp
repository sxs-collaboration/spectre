// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/SliceIterator.hpp"
#include "Domain/Structure/Element.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/Numeric.hpp"

namespace Limiters::Minmod_detail {

MinmodResult tvb_corrected_minmod(const double a, const double b,
                                  const double c, const double tvb_scale) {
  if (fabs(a) <= tvb_scale) {
    return {a, false};
  }
  if ((std::signbit(a) == std::signbit(b)) and
      (std::signbit(a) == std::signbit(c))) {
    // The if/else group below could be more simply written as
    //   std::copysign(std::min({fabs(a), fabs(b), fabs(c)}), a);
    // however, by separating different cases, we gain the ability to
    // distinguish whether or not the limiter activated.
    if (fabs(a) <= fabs(b) and fabs(a) <= fabs(c)) {
      return {a, false};
    } else {
      return {std::copysign(std::min(fabs(b), fabs(c)), a), true};
    }
  } else {
    return {0.0, true};
  }
}

template <size_t VolumeDim>
BufferWrapper<VolumeDim>::BufferWrapper(const Mesh<VolumeDim>& mesh)
    : volume_and_slice_buffer_and_indices_(
          ::volume_and_slice_indices(mesh.extents())),
      volume_and_slice_indices(volume_and_slice_buffer_and_indices_.second) {
  const size_t half_number_boundary_points = alg::accumulate(
      alg::iota(std::array<size_t, VolumeDim>{{}}, 0_st), 0_st,
      [&mesh](const size_t state, const size_t d) {
        return state + mesh.slice_away(d).number_of_grid_points();
      });
  contiguous_boundary_buffer_ =
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      cpp20::make_unique_for_overwrite<double[]>(half_number_boundary_points);
  size_t alloc_offset = 0;
  for (size_t d = 0; d < VolumeDim; ++d) {
    const size_t num_points = mesh.slice_away(d).number_of_grid_points();
    gsl::at(boundary_buffers, d)
        .set_data_ref(contiguous_boundary_buffer_.get() + alloc_offset,
                      num_points);
    alloc_offset += num_points;
  }
}

template <size_t VolumeDim>
double effective_difference_to_neighbor(
    const double u_mean, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size, const size_t dim,
    const Side& side,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes) {
  const auto dir = Direction<VolumeDim>(dim, side);
  ASSERT(element.neighbors().contains(dir),
         "Minmod helper found no neighbors in direction: " << dir);
  const double neighbor_mean = effective_neighbor_means.at(dir);
  const double neighbor_size = effective_neighbor_sizes.at(dir);
  const double distance_factor =
      0.5 * (1.0 + neighbor_size / gsl::at(element_size, dim));
  return (side == Side::Lower ? -1.0 : 1.0) * (neighbor_mean - u_mean) /
         distance_factor;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template class Minmod_detail::BufferWrapper<DIM(data)>;                      \
  template double effective_difference_to_neighbor<DIM(data)>(                 \
      double, const Element<DIM(data)>&, const std::array<double, DIM(data)>&, \
      size_t, const Side&, const DirectionMap<DIM(data), double>&,             \
      const DirectionMap<DIM(data), double>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Limiters::Minmod_detail
