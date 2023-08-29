// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"

#include <algorithm>
#include <array>
#include <limits>

#include "Domain/Structure/Element.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"
#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Limiters::Minmod_detail {

template <size_t VolumeDim>
bool minmod_limited_slopes(
    const gsl::not_null<DataVector*> u_lin_buffer,
    const gsl::not_null<BufferWrapper<VolumeDim>*> buffer,
    const gsl::not_null<double*> u_mean,
    const gsl::not_null<std::array<double, VolumeDim>*> u_limited_slopes,
    const Limiters::MinmodType minmod_type, const double tvb_constant,
    const DataVector& u, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes) {
  // Check that basis is LGL or LG. Note that...
  // - non-Legendre bases may work "out of the box", but are untested
  // - mixed bases may be okay in principle, but are untested
  ASSERT(mesh.basis() ==
             make_array<VolumeDim>(SpatialDiscretization::Basis::Legendre),
         "Unsupported basis: " << mesh);
  ASSERT(mesh.quadrature() ==
                 make_array<VolumeDim>(
                     SpatialDiscretization::Quadrature::GaussLobatto) or
             mesh.quadrature() == make_array<VolumeDim>(
                                      SpatialDiscretization::Quadrature::Gauss),
         "Unsupported quadrature: " << mesh);

  const double tvb_scale = [&tvb_constant, &element_size]() {
    const double max_h =
        *std::max_element(element_size.begin(), element_size.end());
    return tvb_constant * square(max_h);
  }();

  // Results from SpECTRE paper (https://arxiv.org/abs/1609.00098) used a
  // max_slope_factor a factor of 2.0 too small, so that LambdaPi1 behaved
  // like MUSCL, and MUSCL was even more dissipative.
  const double max_slope_factor =
      (minmod_type == Limiters::MinmodType::Muscl) ? 1.0 : 2.0;

  *u_mean = mean_value(u, mesh);

  const auto difference_to_neighbor =
      [&u_mean, &element, &element_size, &effective_neighbor_means,
       &effective_neighbor_sizes](const size_t dim, const Side& side) {
        return effective_difference_to_neighbor(
            *u_mean, element, element_size, dim, side, effective_neighbor_means,
            effective_neighbor_sizes);
      };

  // The LambdaPiN limiter calls a simple troubled-cell indicator to avoid
  // limiting solutions that appear smooth:
  if (minmod_type == Limiters::MinmodType::LambdaPiN) {
    const bool u_needs_limiting = Tci::tvb_minmod_indicator(
        buffer, tvb_constant, u, mesh, element, element_size,
        effective_neighbor_means, effective_neighbor_sizes);

    if (not u_needs_limiting) {
      // Skip the limiting step for this tensor component
#ifdef SPECTRE_DEBUG
      *u_mean = std::numeric_limits<double>::signaling_NaN();
      *u_limited_slopes =
          make_array<VolumeDim>(std::numeric_limits<double>::signaling_NaN());
#endif  // ifdef SPECTRE_DEBUG
      return false;
    }
  }  // end if LambdaPiN

  // If the LambdaPiN check did not skip the limiting, then proceed as normal
  // to determine whether the slopes need to be reduced.
  //
  // Note that we expect the Muscl and LambdaPi1 limiters to linearize the
  // solution whether or not the slope needed reduction. To permit this
  // linearization, we always return (by reference) the slopes when these
  // limiters are in use. In contrast, for LambdaPiN, we only return the slopes
  // when they do in fact need to be reduced.
  bool slopes_need_reducing = false;

  linearize(u_lin_buffer, u, mesh);
  for (size_t d = 0; d < VolumeDim; ++d) {
    const bool has_lower_neighbors =
        element.neighbors().contains(Direction<VolumeDim>(d, Side::Lower));
    const bool has_upper_neighbors =
        element.neighbors().contains(Direction<VolumeDim>(d, Side::Upper));
    // If there are no neighbors on either side, then there isn't enough data
    // to apply the limiter. Skip this case.
    if (UNLIKELY(not has_lower_neighbors and not has_upper_neighbors)) {
      continue;
    }

    auto& boundary_buffer_d = gsl::at(buffer->boundary_buffers, d);
    const auto& volume_and_slice_indices_d =
        gsl::at(buffer->volume_and_slice_indices, d);

    // Compute slope of the linearized U by finite differencing across the
    // first and last grid points. When using Gauss points, we also need to
    // compute the the distance separating these grid points as it will be
    // less than 2.0.
    const double u_lower = mean_value_on_boundary(
        &boundary_buffer_d, volume_and_slice_indices_d.first, *u_lin_buffer,
        mesh, d, Side::Lower);
    const double u_upper = mean_value_on_boundary(
        &boundary_buffer_d, volume_and_slice_indices_d.second, *u_lin_buffer,
        mesh, d, Side::Upper);
    const double first_to_last_distance =
        2.0 *
        (mesh.quadrature(d) == SpatialDiscretization::Quadrature::GaussLobatto
             ? 1.0
             : fabs(Spectral::collocation_points(mesh.slice_through(d))[0]));
    const double local_slope = (u_upper - u_lower) / first_to_last_distance;

    // If one side is an external boundary, we can't define a mean-to-mean
    // difference on that side. We reuse the value of the difference from the
    // internal side, so that the external side has no effect on the result of
    // the minmod function. One alternative implementation would be to use a
    // two-argument minmod function without any arguments corresponding to the
    // external side, but this requires writing more code. Note that we already
    // excluded above the case where both sides are external boundaries.
    //
    // For the effective slopes to neighboring elements, we don't care about the
    // grid point distribution, only the element's width in logical coordinates,
    // which will always be 2.0.
    double lower_slope = 0.0;
    double upper_slope = 0.0;
    if (LIKELY(has_lower_neighbors)) {
      lower_slope = 0.5 * difference_to_neighbor(d, Side::Lower);
      upper_slope =
          (has_upper_neighbors ? 0.5 * difference_to_neighbor(d, Side::Upper)
                               : lower_slope);
    } else {
      upper_slope = 0.5 * difference_to_neighbor(d, Side::Upper);
      lower_slope = upper_slope;  // no lower neighbors in this branch
    }

    const MinmodResult result =
        tvb_corrected_minmod(local_slope, max_slope_factor * upper_slope,
                             max_slope_factor * lower_slope, tvb_scale);
    gsl::at(*u_limited_slopes, d) = result.value;
    if (result.activated) {
      slopes_need_reducing = true;
    }
  }

#ifdef SPECTRE_DEBUG
  // Guard against incorrect use of returned (by reference) slopes in a
  // LambdaPiN limiter, by setting these to NaN when they should not be used.
  if (minmod_type == Limiters::MinmodType::LambdaPiN and
      not slopes_need_reducing) {
    *u_mean = std::numeric_limits<double>::signaling_NaN();
    *u_limited_slopes =
        make_array<VolumeDim>(std::numeric_limits<double>::signaling_NaN());
  }
#endif  // ifdef SPECTRE_DEBUG

  return slopes_need_reducing;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template bool minmod_limited_slopes<DIM(data)>(                  \
      const gsl::not_null<DataVector*>,                            \
      const gsl::not_null<BufferWrapper<DIM(data)>*>,              \
      const gsl::not_null<double*>,                                \
      const gsl::not_null<std::array<double, DIM(data)>*>,         \
      const Limiters::MinmodType, const double, const DataVector&, \
      const Mesh<DIM(data)>&, const Element<DIM(data)>&,           \
      const std::array<double, DIM(data)>&,                        \
      const DirectionMap<DIM(data), double>&,                      \
      const DirectionMap<DIM(data), double>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Limiters::Minmod_detail
