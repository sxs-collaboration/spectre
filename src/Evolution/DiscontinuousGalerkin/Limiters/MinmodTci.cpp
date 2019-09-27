// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"

#include <algorithm>
#include <array>
#include <utility>

#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Limiters {
namespace Minmod_detail {

template <size_t VolumeDim>
bool troubled_cell_indicator(
    const gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const double tvbm_constant, const DataVector& u,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes,
    const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                               gsl::span<std::pair<size_t, size_t>>>,
                     VolumeDim>& volume_and_slice_indices) noexcept {
  const double tvbm_scale = [&tvbm_constant, &element_size ]() noexcept {
    const double max_h =
        *std::max_element(element_size.begin(), element_size.end());
    return tvbm_constant * square(max_h);
  }
  ();
  const double u_mean = mean_value(u, mesh);

  const auto difference_to_neighbor = [
    &u_mean, &element, &element_size, &effective_neighbor_means, &
    effective_neighbor_sizes
  ](const size_t dim, const Side& side) noexcept {
    return effective_difference_to_neighbor(
        u_mean, element, element_size, effective_neighbor_means,
        effective_neighbor_sizes, dim, side);
  };

  for (size_t d = 0; d < VolumeDim; ++d) {
    const double u_lower = mean_value_on_boundary(
        &(gsl::at(*boundary_buffer, d)),
        gsl::at(volume_and_slice_indices, d).first, u, mesh, d, Side::Lower);
    const double u_upper = mean_value_on_boundary(
        &(gsl::at(*boundary_buffer, d)),
        gsl::at(volume_and_slice_indices, d).second, u, mesh, d, Side::Upper);
    const double diff_lower = difference_to_neighbor(d, Side::Lower);
    const double diff_upper = difference_to_neighbor(d, Side::Upper);

    // Results from SpECTRE paper (https://arxiv.org/abs/1609.00098) used
    // minmod_tvbm(..., 0.0), rather than minmod_tvbm(..., tvbm_scale)
    const bool activated_lower =
        minmod_tvbm(u_mean - u_lower, diff_lower, diff_upper, tvbm_scale)
            .activated;
    const bool activated_upper =
        minmod_tvbm(u_upper - u_mean, diff_lower, diff_upper, tvbm_scale)
            .activated;
    if (activated_lower or activated_upper) {
      return true;
    }
  }
  return false;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template bool troubled_cell_indicator<DIM(data)>(                          \
      const gsl::not_null<std::array<DataVector, DIM(data)>*>, const double, \
      const DataVector&, const Element<DIM(data)>&, const Mesh<DIM(data)>&,  \
      const std::array<double, DIM(data)>&,                                  \
      const DirectionMap<DIM(data), double>&,                                \
      const DirectionMap<DIM(data), double>&,                                \
      const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,       \
                                 gsl::span<std::pair<size_t, size_t>>>,      \
                       DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Minmod_detail
}  // namespace Limiters
