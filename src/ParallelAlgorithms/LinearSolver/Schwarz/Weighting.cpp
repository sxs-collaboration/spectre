// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Schwarz/Weighting.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Hypercube.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"

namespace LinearSolver::Schwarz {

DataVector extruding_weight(const DataVector& logical_coords,
                            const double width, const Side& side) noexcept {
  const double sign = side == Side::Lower ? -1. : 1.;
  return 0.5 + 0.5 * sign -
         sign * smoothstep<2>(sign - width, sign + width, logical_coords);
}

namespace {
void apply_element_weight(const gsl::not_null<Scalar<DataVector>*> weight,
                          const DataVector& logical_coords, const double width,
                          const bool has_overlap_lower,
                          const bool has_overlap_upper) noexcept {
  if (has_overlap_lower and has_overlap_upper) {
    get(*weight) *= extruding_weight(logical_coords, width, Side::Lower) +
                    extruding_weight(logical_coords, width, Side::Upper) - 1.;
  } else if (has_overlap_lower) {
    get(*weight) *= extruding_weight(logical_coords, width, Side::Lower);
  } else if (has_overlap_upper) {
    get(*weight) *= extruding_weight(logical_coords, width, Side::Upper);
  }
}
}  // namespace

template <size_t Dim>
void element_weight(
    const gsl::not_null<Scalar<DataVector>*> weight,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& logical_coords,
    const std::array<double, Dim>& overlap_widths,
    const std::unordered_set<Direction<Dim>>& external_boundaries) noexcept {
  destructive_resize_components(weight, logical_coords.begin()->size());
  get(*weight) = 1.;
  for (size_t d = 0; d < Dim; ++d) {
    ASSERT(gsl::at(overlap_widths, d) > 0,
           "Don't try to apply weighting when the overlap has zero width.");
    apply_element_weight(
        weight, logical_coords.get(d), gsl::at(overlap_widths, d),
        external_boundaries.find(Direction<Dim>{d, Side::Lower}) ==
            external_boundaries.end(),
        external_boundaries.find(Direction<Dim>{d, Side::Upper}) ==
            external_boundaries.end());
  }
}

template <size_t Dim>
Scalar<DataVector> element_weight(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& logical_coords,
    const std::array<double, Dim>& overlap_widths,
    const std::unordered_set<Direction<Dim>>& external_boundaries) noexcept {
  Scalar<DataVector> weight{logical_coords.begin()->size()};
  element_weight(make_not_null(&weight), logical_coords, overlap_widths,
                 external_boundaries);
  return weight;
}

DataVector intruding_weight(const DataVector& logical_coords,
                            const double width, const Side& side) noexcept {
  const double sign = side == Side::Lower ? -1. : 1.;
  return extruding_weight(logical_coords - sign * 2., width, opposite(side));
}

namespace {
size_t dim_in_volume(const size_t dim_in_slice,
                     const size_t sliced_dim) noexcept {
  return dim_in_slice >= sliced_dim ? (dim_in_slice + 1) : dim_in_slice;
}
}  // namespace

template <size_t Dim>
void intruding_weight(
    const gsl::not_null<Scalar<DataVector>*> weight,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& logical_coords,
    const Direction<Dim>& direction,
    const std::array<double, Dim>& overlap_widths,
    const size_t num_intruding_overlaps,
    const std::unordered_set<Direction<Dim>>& external_boundaries) noexcept {
  static_assert(Dim > 0 and Dim <= 3,
                "This function supports one, two and three dimensions.");
  ASSERT(gsl::at(overlap_widths, direction.dimension()) > 0,
         "Don't try to apply weighting when the overlap has zero width.");
  if constexpr (Dim == 1) {
    get(*weight) =
        intruding_weight(logical_coords.get(direction.dimension()),
                         gsl::at(overlap_widths, direction.dimension()),
                         direction.side()) /
        num_intruding_overlaps;
  } else {
    const size_t num_points = logical_coords.begin()->size();
    destructive_resize_components(weight, num_points);
    get(*weight) = 1.;
    const auto has_overlap = [&external_boundaries](const size_t dimension,
                                                    const Side side) noexcept {
      return external_boundaries.find(Direction<Dim>{dimension, side}) ==
             external_boundaries.end();
    };
    // Apply weighting perpendicular to the overlap direction
    for (size_t d = 0; (d == direction.dimension() ? ++d : d) < Dim; ++d) {
      ASSERT(gsl::at(overlap_widths, d) > 0,
             "Don't try to apply weighting when the overlap has zero width.");
      apply_element_weight(
          weight, logical_coords.get(d), gsl::at(overlap_widths, d),
          has_overlap(d, Side::Lower), has_overlap(d, Side::Upper));
    }
    // Add contributions from the corners and edges of the subdomain
    // These contributions account for the corner- and edge-neighbors not being
    // part of the subdomain, and thus not contributing weights. To retain
    // conservation, we add the missing weights to the intruding overlaps here.
    // See the function documentation for details.
    const auto skip_corner =
        [&direction, &has_overlap](const Vertex<Dim - 1>& corner) noexcept {
          // Skip corners to external boundaries
          for (size_t d = 0; d < Dim - 1; ++d) {
            if (not has_overlap(dim_in_volume(d, direction.dimension()),
                                corner.side_in_parent_dimension(d))) {
              return true;
            }
          }
          return false;
        };
    Scalar<DataVector> corner_weight{num_points};
    for (const auto corner : VertexIterator<Dim - 1>{}) {
      if (skip_corner(corner)) {
        continue;
      }
      get(corner_weight) = 1.;
      for (size_t d = 0; d < Dim - 1; ++d) {
        const size_t d_vol = dim_in_volume(d, direction.dimension());
        get(corner_weight) *= intruding_weight(
            logical_coords.get(d_vol), gsl::at(overlap_widths, d_vol),
            corner.side_in_parent_dimension(d));
      }
      // Divide equally between all face-neighbors that share this corner
      get(*weight) += get(corner_weight) / Dim;
    }
    if constexpr (Dim == 3) {
      Scalar<DataVector> edge_weight{num_points};
      for (const auto edge : EdgeIterator<Dim - 1>{}) {
        const size_t d_perp =
            dim_in_volume(dim_in_volume(0, edge.dimension_in_parent()),
                          direction.dimension());
        // Skip edges to external boundaries
        if (not has_overlap(d_perp, edge.side())) {
          continue;
        }
        get(edge_weight) = 1.;
        const size_t d_edge =
            dim_in_volume(edge.dimension_in_parent(), direction.dimension());
        apply_element_weight(
            make_not_null(&edge_weight), logical_coords.get(d_edge),
            gsl::at(overlap_widths, d_edge), has_overlap(d_edge, Side::Lower),
            has_overlap(d_edge, Side::Upper));
        get(edge_weight) *=
            intruding_weight(logical_coords.get(d_perp),
                             gsl::at(overlap_widths, d_perp), edge.side());
        // Divide equally between all face-neighbors that share this edge
        get(*weight) += get(edge_weight) / (Dim - 1);
      }
    }
    // Apply weighting along the overlap direction
    get(*weight) *=
        intruding_weight(logical_coords.get(direction.dimension()),
                         gsl::at(overlap_widths, direction.dimension()),
                         direction.side()) /
        num_intruding_overlaps;
  }
}

template <size_t Dim>
Scalar<DataVector> intruding_weight(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& logical_coords,
    const Direction<Dim>& direction,
    const std::array<double, Dim>& overlap_widths,
    const size_t num_intruding_overlaps,
    const std::unordered_set<Direction<Dim>>& external_boundaries) noexcept {
  Scalar<DataVector> weight{logical_coords.begin()->size()};
  intruding_weight(make_not_null(&weight), logical_coords, direction,
                   overlap_widths, num_intruding_overlaps, external_boundaries);
  return weight;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data)                                                \
  template void element_weight(                                             \
      gsl::not_null<Scalar<DataVector>*> weight,                            \
      const tnsr::I<DataVector, DIM(data), Frame::ElementLogical>&          \
          logical_coords,                                                   \
      const std::array<double, DIM(data)>& overlap_widths,                  \
      const std::unordered_set<Direction<DIM(data)>>& external_boundaries); \
  template Scalar<DataVector> element_weight(                               \
      const tnsr::I<DataVector, DIM(data), Frame::ElementLogical>&          \
          logical_coords,                                                   \
      const std::array<double, DIM(data)>& overlap_widths,                  \
      const std::unordered_set<Direction<DIM(data)>>& external_boundaries); \
  template void intruding_weight(                                           \
      gsl::not_null<Scalar<DataVector>*> weight,                            \
      const tnsr::I<DataVector, DIM(data), Frame::ElementLogical>&          \
          logical_coords,                                                   \
      const Direction<DIM(data)>& direction,                                \
      const std::array<double, DIM(data)>& overlap_widths,                  \
      size_t num_intruding_overlaps,                                        \
      const std::unordered_set<Direction<DIM(data)>>& external_boundaries); \
  template Scalar<DataVector> intruding_weight(                             \
      const tnsr::I<DataVector, DIM(data), Frame::ElementLogical>&          \
          logical_coords,                                                   \
      const Direction<DIM(data)>& direction,                                \
      const std::array<double, DIM(data)>& overlap_widths,                  \
      size_t num_intruding_overlaps,                                        \
      const std::unordered_set<Direction<DIM(data)>>& external_boundaries);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace LinearSolver::Schwarz
