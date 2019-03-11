// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

namespace SlopeLimiters {
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
    const gsl::not_null<DataVector*> u_lin_buffer,
    const gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const SlopeLimiters::MinmodType& minmod_type, const double tvbm_constant,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const FixedHashMap<
        maximum_number_of_neighbors(VolumeDim),
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        gsl::not_null<const double*>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_tensor_begin,
    const FixedHashMap<
        maximum_number_of_neighbors(VolumeDim),
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        std::array<double, VolumeDim>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_sizes,
    const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                               gsl::span<std::pair<size_t, size_t>>>,
                     VolumeDim>& volume_and_slice_indices) noexcept {
  // True if the mesh is linear-order in every direction
  const bool mesh_is_linear = (mesh.extents() == Index<VolumeDim>(2));
  const bool minmod_type_is_linear =
      (minmod_type != SlopeLimiters::MinmodType::LambdaPiN);
  const bool using_linear_limiter_on_non_linear_mesh =
      minmod_type_is_linear and not mesh_is_linear;

  const double tvbm_scale = [&tvbm_constant, &element_size ]() noexcept {
    const double max_h =
        *std::max_element(element_size.begin(), element_size.end());
    return tvbm_constant * square(max_h);
  }
  ();

  // Results from SpECTRE paper (https://arxiv.org/abs/1609.00098) used a
  // max_slope_factor a factor of 2.0 too small, so that LambdaPi1 behaved
  // like MUSCL, and MUSCL was even more dissipative.
  const double max_slope_factor =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 1.0 : 2.0;

  // In each direction, average the size of all different neighbors in that
  // direction. Note that only the component of neighor_size that is normal
  // to the face is needed (and, therefore, computed). Note that this average
  // does not depend on the solution on the neighboring elements, so could be
  // precomputed outside of `limit_one_tensor`. Changing the code to
  // precompute the average may or may not be a measurable optimization.
  const auto effective_neighbor_sizes =
      [&neighbor_sizes, &element ]() noexcept {
    DirectionMap<VolumeDim, double> result;
    for (const auto& dir : Direction<VolumeDim>::all_directions()) {
      const auto& externals = element.external_boundaries();
      const bool neighbors_in_this_dir =
          (externals.find(dir) == externals.end());
      if (neighbors_in_this_dir) {
        const double effective_neighbor_size =
            [&neighbor_sizes, &dir, &element ]() noexcept {
          const size_t dim = dir.dimension();
          const auto& neighbor_ids = element.neighbors().at(dir).ids();
          double size_accumulate = 0.0;
          for (const auto& id : neighbor_ids) {
            size_accumulate +=
                gsl::at(neighbor_sizes.at(std::make_pair(dir, id)), dim);
          }
          return size_accumulate / neighbor_ids.size();
        }
        ();
        result.insert(std::make_pair(dir, effective_neighbor_size));
      }
    }
    return result;
  }
  ();

  bool some_component_was_limited = false;
  for (auto iter = tensor_begin.get(); iter != tensor_end.get(); ++iter) {
    DataVector& u = *iter;
    const double u_mean = mean_value(u, mesh);

    const auto iter_offset = std::distance(tensor_begin.get(), iter);
    // In each direction, average the mean of the `iter` tensor component over
    // all different neighbors in that direction. This produces one effective
    // neighbor per direction.
    const auto effective_neighbor_means =
        [&neighbor_tensor_begin, &element, &iter_offset ]() noexcept {
      DirectionMap<VolumeDim, double> result;
      for (const auto& dir : Direction<VolumeDim>::all_directions()) {
        const auto& externals = element.external_boundaries();
        const bool neighbors_in_this_dir =
            (externals.find(dir) == externals.end());
        if (neighbors_in_this_dir) {
          const double effective_neighbor_mean = [
            &neighbor_tensor_begin, &dir, &element, &iter_offset
          ]() noexcept {
            const auto& neighbor_ids = element.neighbors().at(dir).ids();
            double mean_accumulate = 0.0;
            for (const auto& id : neighbor_ids) {
              // clang-tidy: do not use pointer arithmetic
              mean_accumulate +=
                  *(neighbor_tensor_begin.at(std::make_pair(dir, id)).get() +
                    iter_offset);  // NOLINT
            }
            return mean_accumulate / neighbor_ids.size();
          }
          ();
          result.insert(std::make_pair(dir, effective_neighbor_mean));
        }
      }
      return result;
    }
    ();

    const auto difference_to_neighbor = [
      &u_mean, &element, &element_size, &effective_neighbor_means, &
      effective_neighbor_sizes
    ](const size_t dim, const Side& side) noexcept {
      const auto& externals = element.external_boundaries();
      const auto dir = Direction<VolumeDim>(dim, side);
      const bool has_neighbors = (externals.find(dir) == externals.end());
      if (has_neighbors) {
        const double neighbor_size = effective_neighbor_sizes.at(dir);
        const double neighbor_mean = effective_neighbor_means.at(dir);

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
            0.5 * (1.0 + neighbor_size / gsl::at(element_size, dim));
        return (side == Side::Lower ? -1.0 : 1.0) * (neighbor_mean - u_mean) /
               distance_factor;
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
            mean_value_on_boundary(&(gsl::at(*boundary_buffer, d)),
                                   gsl::at(volume_and_slice_indices, d).first,
                                   u, mesh, d, Side::Lower);
        const double u_upper =
            mean_value_on_boundary(&(gsl::at(*boundary_buffer, d)),
                                   gsl::at(volume_and_slice_indices, d).second,
                                   u, mesh, d, Side::Upper);
        const double diff_lower = difference_to_neighbor(d, Side::Lower);
        const double diff_upper = difference_to_neighbor(d, Side::Upper);

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

    linearize(u_lin_buffer, u, mesh);
    bool reduce_slope = false;
    auto u_limited_slopes = make_array<VolumeDim>(0.0);

    for (size_t d = 0; d < VolumeDim; ++d) {
      const double u_lower =
          mean_value_on_boundary(&(gsl::at(*boundary_buffer, d)),
                                 gsl::at(volume_and_slice_indices, d).first,
                                 *u_lin_buffer, mesh, d, Side::Lower);
      const double u_upper =
          mean_value_on_boundary(&(gsl::at(*boundary_buffer, d)),
                                 gsl::at(volume_and_slice_indices, d).second,
                                 *u_lin_buffer, mesh, d, Side::Upper);

      // Divide by element's width (2.0 in logical coordinates) to get a slope
      const double local_slope = 0.5 * (u_upper - u_lower);
      const double upper_slope = 0.5 * difference_to_neighbor(d, Side::Upper);
      const double lower_slope = 0.5 * difference_to_neighbor(d, Side::Lower);

      const MinmodResult& result =
          minmod_tvbm(local_slope, max_slope_factor * upper_slope,
                      max_slope_factor * lower_slope, tvbm_scale);
      gsl::at(u_limited_slopes, d) = result.value;
      if (result.activated) {
        reduce_slope = true;
      }
    }

    if (reduce_slope or using_linear_limiter_on_non_linear_mesh) {
      u = u_mean;
      for (size_t d = 0; d < VolumeDim; ++d) {
        u += logical_coords.get(d) * gsl::at(u_limited_slopes, d);
      }
      some_component_was_limited = true;
    }
  }

  return some_component_was_limited;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template bool limit_one_tensor<DIM(data)>(                              \
      const gsl::not_null<DataVector*>, const gsl::not_null<DataVector*>, \
      const gsl::not_null<DataVector*>,                                   \
      const gsl::not_null<std::array<DataVector, DIM(data)>*>,            \
      const SlopeLimiters::MinmodType&, const double,                     \
      const Element<DIM(data)>&, const Mesh<DIM(data)>&,                  \
      const tnsr::I<DataVector, DIM(data), Frame::Logical>&,              \
      const std::array<double, DIM(data)>&,                               \
      const FixedHashMap<                                                 \
          maximum_number_of_neighbors(DIM(data)),                         \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,          \
          gsl::not_null<const double*>,                                   \
          boost::hash<                                                    \
              std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>&,   \
      const FixedHashMap<                                                 \
          maximum_number_of_neighbors(DIM(data)),                         \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,          \
          std::array<double, DIM(data)>,                                  \
          boost::hash<                                                    \
              std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>&,   \
      const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,    \
                                 gsl::span<std::pair<size_t, size_t>>>,   \
                       DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace Minmod_detail
}  // namespace SlopeLimiters
