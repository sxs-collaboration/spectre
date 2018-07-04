// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeArray.hpp"

// IWYU pragma: no_include <ostream>

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

// Implements the minmod limiter for one Tensor<DataVector> at a time.
template <size_t VolumeDim>
bool limit_one_tensor(
    const gsl::not_null<DataVector*> tensor_begin,
    const gsl::not_null<DataVector*> tensor_end,
    const std::unordered_map<domain::Direction<VolumeDim>,
                             gsl::not_null<const double*>>&
        neighbor_tensor_begin,
    const SlopeLimiters::MinmodType& minmod_type, const double tvbm_constant,
    const domain::Element<VolumeDim>& element,
    const domain::Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const tnsr::I<double, VolumeDim>& element_size,
    const std::unordered_map<domain::Direction<VolumeDim>,
                             tnsr::I<double, VolumeDim>>&
        neighbor_sizes) noexcept {
  // True if the mesh is linear-order in every direction
  const bool mesh_is_linear = (mesh.extents() == Index<VolumeDim>(2));

  const double tvbm_scale = [&tvbm_constant, &element_size ]() noexcept {
    double max_h_sqr = 0.0;
    for (size_t d = 0; d < VolumeDim; ++d) {
      max_h_sqr = std::max(max_h_sqr, square(element_size.get(d)));
    }
    return tvbm_constant * max_h_sqr;
  }
  ();

  // Results from SpECTRE paper (https://arxiv.org/abs/1609.00098) used a
  // max_slope_factor a factor of 2.0 too small, so that LambdaPi1 behaved
  // like MUSCL, and MUSCL was even more dissipative.
  const double max_slope_factor =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 1.0 : 2.0;

  bool some_component_was_limited = false;
  for (auto iter = tensor_begin.get(); iter != tensor_end.get(); ++iter) {
    DataVector& u = *iter;
    const double u_mean = mean_value(u, mesh);

    const auto iter_offset = std::distance(tensor_begin.get(), iter);
    const auto difference_to_neighbor = [
      &u_mean, &neighbor_tensor_begin, &element, &element_size, &neighbor_sizes,
      &iter_offset
    ](const size_t dim, const domain::Side& side) noexcept {
      const auto& externals = element.external_boundaries();
      const auto dir = domain::Direction<VolumeDim>(dim, side);
      const bool has_neighbors = (externals.find(dir) == externals.end());
      if (has_neighbors) {
        // Compute an effective element-center-to-neighbor-center distance
        // that accounts for the possibility of different refinement levels
        // or discontinuous maps (e.g., at Block boundaries). Treated naively,
        // these domain features can make a smooth solution appear to be
        // non-smooth in the logical coordinates, which could potentially lead
        // to the limiter triggering erroneously. This effective distance is
        // used to scale the difference in the means, so that a linear function
        // at a refinement or Block boundary will still appear smooth to the
        // limiter. The factor is normalized to be 1.0 on a uniform grid.
        // Note that this is not "by the book" Minmod, but an attempt to
        // generalize Minmod to work on non-uniform grids.
        const double distance_factor =
            0.5 *
            (1.0 + neighbor_sizes.at(dir).get(dim) / element_size.get(dim));
        const double neighbor_mean =
            // clang-tidy: do not use pointer arithmetic
            *(neighbor_tensor_begin.at(dir).get() + iter_offset);  // NOLINT
        return (side == domain::Side::Lower ? -1.0 : 1.0) *
               (neighbor_mean - u_mean) / distance_factor;
      } else {
        return 0.0;
      }
    };

    // The LambdaPiN limiter allows high-order solutions to escape limiting if
    // the boundary values are not too different from the mean value:
    if (minmod_type == SlopeLimiters::MinmodType::LambdaPiN) {
      bool u_needs_limiting = false;
      for (size_t d = 0; d < VolumeDim; ++d) {
        const double u_lower =
            mean_value_on_boundary(u, mesh, d, domain::Side::Lower);
        const double u_upper =
            mean_value_on_boundary(u, mesh, d, domain::Side::Upper);
        const double diff_lower =
            difference_to_neighbor(d, domain::Side::Lower);
        const double diff_upper =
            difference_to_neighbor(d, domain::Side::Upper);

        // Results from SpECTRE paper (https://arxiv.org/abs/1609.00098) used
        // minmod_tvbm(..., 0.0), rather than minmod_tvbm(..., tvbm_scale)
        const double v_lower =
            u_mean -
            minmod_tvbm(u_mean - u_lower, diff_lower, diff_upper, tvbm_scale)
                .value;
        const double v_upper =
            u_mean +
            minmod_tvbm(u_upper - u_mean, diff_lower, diff_upper, tvbm_scale)
                .value;
        // Value of epsilon from Hesthaven & Warburton, Chapter 5, in the
        // SlopeLimitN.m code sample.
        const double eps = 1.e-8;
        if (fabs(v_lower - u_lower) > eps or fabs(v_upper - u_upper) > eps) {
          u_needs_limiting = true;
          break;
        }
      }

      if (not u_needs_limiting) {
        // Skip the limiting step for this tensor component
        continue;
      }
    }

    const DataVector u_lin = linearize(u, mesh);
    bool this_component_was_limited = false;
    auto u_limited_slopes = make_array<VolumeDim>(0.0);

    for (size_t d = 0; d < VolumeDim; ++d) {
      const double u_lower =
          mean_value_on_boundary(u_lin, mesh, d, domain::Side::Lower);
      const double u_upper =
          mean_value_on_boundary(u_lin, mesh, d, domain::Side::Upper);

      // Divide by element's width (2.0 in logical coordinates) to get a slope
      const double local_slope = 0.5 * (u_upper - u_lower);
      const double upper_slope =
          0.5 * difference_to_neighbor(d, domain::Side::Upper);
      const double lower_slope =
          0.5 * difference_to_neighbor(d, domain::Side::Lower);

      const MinmodResult& result =
          minmod_tvbm(local_slope, max_slope_factor * upper_slope,
                      max_slope_factor * lower_slope, tvbm_scale);
      gsl::at(u_limited_slopes, d) = result.value;
      if (result.activated or not mesh_is_linear) {
        this_component_was_limited = true;
      }
    }

    // Optimization: if the mesh is linear (so that linearization is a no-op),
    // and the limiter did not request a reduced slope, then there is no need to
    // overwite u = u_mean + (new limited slope in x) * x + etc., in the
    // loop below, so skip it.
    if (mesh_is_linear and not this_component_was_limited) {
      continue;
    }

    u = u_mean;
    for (size_t d = 0; d < VolumeDim; ++d) {
      u += logical_coords.get(d) * gsl::at(u_limited_slopes, d);
    }
    some_component_was_limited =
        (some_component_was_limited or this_component_was_limited);
  }

  return some_component_was_limited;
}

// Explicit instantiations
template bool limit_one_tensor<1>(
    const gsl::not_null<DataVector*>, const gsl::not_null<DataVector*>,
    const std::unordered_map<domain::Direction<1>,
                             gsl::not_null<const double*>>&,
    const SlopeLimiters::MinmodType&, const double, const domain::Element<1>&,
    const domain::Mesh<1>&, const tnsr::I<DataVector, 1, Frame::Logical>&,
    const tnsr::I<double, 1>&,
    const std::unordered_map<domain::Direction<1>,
                             tnsr::I<double, 1>>&) noexcept;
template bool limit_one_tensor<2>(
    const gsl::not_null<DataVector*>, const gsl::not_null<DataVector*>,
    const std::unordered_map<domain::Direction<2>,
                             gsl::not_null<const double*>>&,
    const SlopeLimiters::MinmodType&, const double, const domain::Element<2>&,
    const domain::Mesh<2>&, const tnsr::I<DataVector, 2, Frame::Logical>&,
    const tnsr::I<double, 2>&,
    const std::unordered_map<domain::Direction<2>,
                             tnsr::I<double, 2>>&) noexcept;
template bool limit_one_tensor<3>(
    const gsl::not_null<DataVector*>, const gsl::not_null<DataVector*>,
    const std::unordered_map<domain::Direction<3>,
                             gsl::not_null<const double*>>&,
    const SlopeLimiters::MinmodType&, const double, const domain::Element<3>&,
    const domain::Mesh<3>&, const tnsr::I<DataVector, 3, Frame::Logical>&,
    const tnsr::I<double, 3>&,
    const std::unordered_map<domain::Direction<3>,
                             tnsr::I<double, 3>>&) noexcept;
}  // namespace Minmod_detail

SlopeLimiters::MinmodType create_from_yaml<SlopeLimiters::MinmodType>::create(
    const Option& options) {
  const std::string minmod_type_read = options.parse_as<std::string>();
  if (minmod_type_read == "LambdaPi1") {
    return SlopeLimiters::MinmodType::LambdaPi1;
  } else if (minmod_type_read == "LambdaPiN") {
    return SlopeLimiters::MinmodType::LambdaPiN;
  } else if (minmod_type_read == "Muscl") {
    return SlopeLimiters::MinmodType::Muscl;
  }
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << minmod_type_read
                                     << "\" to MinmodType. Expected one of: "
                                        "{LambdaPi1, LambdaPiN, Muscl}.");
}
