// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/Element.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Limiters::Tci {

template <size_t VolumeDim>
bool tvb_minmod_indicator(
    const gsl::not_null<Minmod_detail::BufferWrapper<VolumeDim>*> buffer,
    const double tvb_constant, const DataVector& u, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes) noexcept {
  // Check that basis is LGL or LG. Note that...
  // - non-Legendre bases may work "out of the box", but are untested
  // - mixed bases may be okay in principle, but are untested
  ASSERT(mesh.basis() == make_array<VolumeDim>(Spectral::Basis::Legendre),
         "Unsupported basis: " << mesh);
  ASSERT(mesh.quadrature() ==
                 make_array<VolumeDim>(Spectral::Quadrature::GaussLobatto) or
             mesh.quadrature() ==
                 make_array<VolumeDim>(Spectral::Quadrature::Gauss),
         "Unsupported quadrature: " << mesh);

  const double tvb_scale = [&tvb_constant, &element_size]() noexcept {
    const double max_h =
        *std::max_element(element_size.begin(), element_size.end());
    return tvb_constant * square(max_h);
  }();
  const double u_mean = mean_value(u, mesh);

  const auto difference_to_neighbor =
      [&u_mean, &element, &element_size, &effective_neighbor_means,
       &effective_neighbor_sizes](const size_t dim, const Side& side) noexcept {
        return Minmod_detail::effective_difference_to_neighbor(
            u_mean, element, element_size, dim, side, effective_neighbor_means,
            effective_neighbor_sizes);
      };

  for (size_t d = 0; d < VolumeDim; ++d) {
    const bool has_lower_neighbors =
        element.neighbors().contains(Direction<VolumeDim>(d, Side::Lower));
    const bool has_upper_neighbors =
        element.neighbors().contains(Direction<VolumeDim>(d, Side::Upper));
    // If there are no neighbors on either side, then there isn't enough data
    // to even check if the limiter needs to be applied. Skip this case.
    if (UNLIKELY(not has_lower_neighbors and not has_upper_neighbors)) {
      continue;
    }

    double u_lower = 0.;
    double u_upper = 0.;
    auto& boundary_buffers_d = gsl::at(buffer->boundary_buffers, d);

    // This TCI compares mean-to-neighbor vs mean-to-cell-boundary differences.
    // In the case of an LGL mesh, the boundary values can be read off directly,
    // but in the case of a LG mesh, we must interpolate to the boundary.
    if (mesh.quadrature(d) == Spectral::Quadrature::GaussLobatto) {
      const auto& volume_and_slice_indices_d =
          gsl::at(buffer->volume_and_slice_indices, d);
      u_lower = mean_value_on_boundary(&boundary_buffers_d,
                                       volume_and_slice_indices_d.first, u,
                                       mesh, d, Side::Lower);
      u_upper = mean_value_on_boundary(&boundary_buffers_d,
                                       volume_and_slice_indices_d.second, u,
                                       mesh, d, Side::Upper);
    } else {
      // We have Spectral::Quadrature::Gauss, so interpolate to boundary
      const Matrix identity{};
      auto interpolation_matrices = make_array<VolumeDim>(std::cref(identity));
      const auto& matrices =
          Spectral::boundary_interpolation_matrices(mesh.slice_through(d));
      gsl::at(interpolation_matrices, d) = matrices.first;
      apply_matrices(make_not_null(&boundary_buffers_d), interpolation_matrices,
                     u, mesh.extents());
      const auto boundary_mesh = mesh.slice_away(d);
      u_lower = mean_value(boundary_buffers_d, boundary_mesh);

      gsl::at(interpolation_matrices, d) = matrices.second;
      apply_matrices(make_not_null(&boundary_buffers_d), interpolation_matrices,
                     u, mesh.extents());
      u_upper = mean_value(boundary_buffers_d, boundary_mesh);
    }

    // If one side is an external boundary, we can't define a mean-to-mean
    // difference on that side. We reuse the value of the difference from the
    // internal side, so that the external side has no effect on the result of
    // the minmod function. One alternative implementation would be to use a
    // two-argument minmod function without any arguments corresponding to the
    // external side, but this requires writing more code. Note that we already
    // excluded above the case where both sides are external boundaries.
    double diff_lower = 0.0;
    double diff_upper = 0.0;
    if (LIKELY(has_lower_neighbors)) {
      diff_lower = difference_to_neighbor(d, Side::Lower);
      diff_upper = (has_upper_neighbors ? difference_to_neighbor(d, Side::Upper)
                                        : diff_lower);
    } else {
      diff_upper = difference_to_neighbor(d, Side::Upper);
      diff_lower = diff_upper;  // no lower neighbors in this branch
    }

    // Results from SpECTRE paper (https://arxiv.org/abs/1609.00098) used
    // tvb_corrected_minmod(..., 0.0), rather than
    // tvb_corrected_minmod(..., tvb_scale)
    const bool activated_lower =
        Minmod_detail::tvb_corrected_minmod(u_mean - u_lower, diff_lower,
                                            diff_upper, tvb_scale)
            .activated;
    const bool activated_upper =
        Minmod_detail::tvb_corrected_minmod(u_upper - u_mean, diff_lower,
                                            diff_upper, tvb_scale)
            .activated;
    if (activated_lower or activated_upper) {
      return true;
    }
  }
  return false;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                           \
  template bool tvb_minmod_indicator<DIM(data)>(                       \
      const gsl::not_null<Minmod_detail::BufferWrapper<DIM(data)>*>,   \
      const double, const DataVector&, const Mesh<DIM(data)>&,         \
      const Element<DIM(data)>&, const std::array<double, DIM(data)>&, \
      const DirectionMap<DIM(data), double>&,                          \
      const DirectionMap<DIM(data), double>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Limiters::Tci
