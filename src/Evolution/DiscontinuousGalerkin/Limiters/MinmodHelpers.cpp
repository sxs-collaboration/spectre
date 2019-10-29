// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Domain/Element.hpp"             // IWYU pragma: keep
#include "Domain/Mesh.hpp"                // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"

namespace Limiters {
namespace Minmod_detail {

MinmodResult minmod_tvbm(const double a, const double b, const double c,
                         const double tvbm_scale) noexcept {
  if (fabs(a) <= tvbm_scale) {
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
void allocate_buffers(
    const gsl::not_null<std::unique_ptr<double[], decltype(&free)>*>
        contiguous_buffer,
    const gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const Mesh<VolumeDim>& mesh) noexcept {
  const size_t half_number_boundary_points = alg::accumulate(
      alg::iota(std::array<size_t, VolumeDim>{{}}, 0_st),
      0_st, [&mesh](const size_t state, const size_t d) noexcept {
        return state + mesh.slice_away(d).number_of_grid_points();
      });
  contiguous_buffer->reset(static_cast<double*>(
      // clang-tidy incorrectly thinks this is a 0-byte malloc
      // NOLINTNEXTLINE(clang-analyzer-unix.API)
      malloc(sizeof(double) * half_number_boundary_points)));
  size_t alloc_offset = 0;
  for (size_t d = 0; d < VolumeDim; ++d) {
    const size_t num_points = mesh.slice_away(d).number_of_grid_points();
    gsl::at(*boundary_buffer, d)
        .set_data_ref(contiguous_buffer->get() + alloc_offset, num_points);
    alloc_offset += num_points;
  }
}

template <size_t VolumeDim>
void allocate_buffers(
    const gsl::not_null<std::unique_ptr<double[], decltype(&free)>*>
        contiguous_buffer,
    const gsl::not_null<DataVector*> u_lin_buffer,
    const gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const Mesh<VolumeDim>& mesh) noexcept {
  const size_t half_number_boundary_points = alg::accumulate(
      alg::iota(std::array<size_t, VolumeDim>{{}}, 0_st),
      0_st, [&mesh](const size_t state, const size_t d) noexcept {
        return state + mesh.slice_away(d).number_of_grid_points();
      });
  contiguous_buffer->reset(static_cast<double*>(
      malloc(sizeof(double) *
             (mesh.number_of_grid_points() + half_number_boundary_points))));
  size_t alloc_offset = 0;
  u_lin_buffer->set_data_ref(contiguous_buffer->get() + alloc_offset,
                             mesh.number_of_grid_points());
  alloc_offset += mesh.number_of_grid_points();
  for (size_t d = 0; d < VolumeDim; ++d) {
    const size_t num_points = mesh.slice_away(d).number_of_grid_points();
    gsl::at(*boundary_buffer, d)
        .set_data_ref(contiguous_buffer->get() + alloc_offset, num_points);
    alloc_offset += num_points;
  }
}

template <size_t VolumeDim>
double effective_difference_to_neighbor(
    const double u_mean, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes,
    const size_t dim, const Side& side) noexcept {
  const auto& externals = element.external_boundaries();
  const auto dir = Direction<VolumeDim>(dim, side);
  const bool has_neighbors = (externals.find(dir) == externals.end());
  if (has_neighbors) {
    const double neighbor_mean = effective_neighbor_means.at(dir);
    const double neighbor_size = effective_neighbor_sizes.at(dir);
    const double distance_factor =
        0.5 * (1.0 + neighbor_size / gsl::at(element_size, dim));
    return (side == Side::Lower ? -1.0 : 1.0) * (neighbor_mean - u_mean) /
           distance_factor;
  } else {
    return 0.0;
  }
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void allocate_buffers<DIM(data)>(                                   \
      const gsl::not_null<std::unique_ptr<double[], decltype(&free)>*>,        \
      const gsl::not_null<std::array<DataVector, DIM(data)>*>,                 \
      const Mesh<DIM(data)>&) noexcept;                                        \
  template void allocate_buffers<DIM(data)>(                                   \
      const gsl::not_null<std::unique_ptr<double[], decltype(&free)>*>,        \
      const gsl::not_null<DataVector*>,                                        \
      const gsl::not_null<std::array<DataVector, DIM(data)>*>,                 \
      const Mesh<DIM(data)>&) noexcept;                                        \
  template double effective_difference_to_neighbor<DIM(data)>(                 \
      double, const Element<DIM(data)>&, const std::array<double, DIM(data)>&, \
      const DirectionMap<DIM(data), double>&,                                  \
      const DirectionMap<DIM(data), double>&, size_t, const Side&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Minmod_detail
}  // namespace Limiters
