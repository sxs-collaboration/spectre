// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/SecondPartialDerivatives.hpp"

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Transpose.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace fd {
namespace {
template <size_t Order, bool UnitStride>
struct ComputeImpl;

// only the 4-th order is implemented for now
template <bool UnitStride>
struct ComputeImpl<4, UnitStride> {
  static constexpr size_t fd_order = 4;

  // The stencil is used where ghost data from two neighbors is needed.
  // It uses 13 points including the point of derivative (the 0th
  // point "o"). We use 1-4 to label the four points distant \delta from the 0th
  // point, 5-6 to label the two diagonal points distant sqrt(2)*\delta from the
  // 0th point, 7-10 to label the four points distant 2*\delta from the 0th
  // point, and 11-12 to label the two diagonal points distant 2*sqrt(2)*\delta
  // from the 0th point. Since the weights associated with the points of the
  // same distance to the 0th point are the same, the ordering with each
  // subgroup above is arbitrary.
  // Schematically, this stencil (with descending diagonal) appears
  // 12  x  8  x  x
  //  x  6  2  x  x
  //  9  3 o0  1  7
  //  x  x  4  5  x
  //  x  x 10  x 11
  SPECTRE_ALWAYS_INLINE static double mixed_second_deriv_special_pointwise(
      const std::array<double, 13>& q, const int stride,
      const std::array<double, 3>& weights) {
    ASSERT(stride == 1, "Only UnitStride is supported" << stride);
    return weights[0] * q[0] +
           weights[1] * (q[1] + q[2] + q[3] + q[4] - q[5] - q[6]) +
           weights[2] * (q[11] + q[12] - q[7] - q[8] - q[9] - q[10]);
  }

  static constexpr std::array<double, 3>
  mixed_second_derivative_special_weights(const double one_over_delta_squared) {
    return {{-1.25 * one_over_delta_squared,
             0.66666666666666667 * one_over_delta_squared,
             0.04166666666666667 * one_over_delta_squared}};
  }

  // This stencil is used in the bulk.
  // 0 - (x+\delta, y+\delta), 1 - (x-\delta, y-\delta), 2 - (x+\delta,
  // y-\delta), 3 - (x-\delta, y+\delta), 4 - (x+2\delta, y+2\delta), 5 -
  // (x-2\delta, y-2\delta), 6 - (x+2\delta, y-2\delta), 7 - (x-2\delta,
  // y+2\delta)
  // 7 x x x 4
  // x 3 x 0 x
  // x x o x x
  // x 1 x 2 x
  // 5 x x x 6
  SPECTRE_ALWAYS_INLINE static double mixed_second_deriv_pointwise(
      const std::array<double, 8>& q, const int stride,
      const std::array<double, 2>& weights) {
    ASSERT(stride == 1, "Only UnitStride is supported" << stride);
    return weights[0] * (q[0] + q[1] - q[2] - q[3]) +
           weights[1] * (-q[4] - q[5] + q[6] + q[7]);
  }

  static constexpr std::array<double, 2> mixed_second_derivative_weights(
      const double one_over_delta_squared) {
    return {{0.33333333333333333 * one_over_delta_squared,
             0.02083333333333333 * one_over_delta_squared}};
  }

  // This stencil is used both in the bulk and the ghost zone.
  // We label the point of derivative as the 0th point. We use
  // 1-2 to label the points distant \delta to the 0th point and 3-4 to label
  // the points distant 2*\delta to the 0th point. Note that this is different
  // from the pointer arithmetic convention in PartialDerivatives.cpp
  // 4 2 0 1 3
  SPECTRE_ALWAYS_INLINE static double pure_second_deriv_pointwise(
      const std::array<double, 5>& q, const int stride,
      const std::array<double, 3>& weights) {
    ASSERT(stride == 1, "Only UnitStride is supported" << stride);
    return weights[0] * q[0] + weights[1] * (q[1] + q[2]) +
           weights[2] * (q[3] + q[4]);
  }

  static constexpr std::array<double, 3> pure_second_derivative_weights(
      const double one_over_delta_squared) {
    return {{-2.5 * one_over_delta_squared,
             1.33333333333333333 * one_over_delta_squared,
             -0.08333333333333333 * one_over_delta_squared}};
  }
};

// Compute the xx, xy derivatives
// x direction points to right_ghost_data; y direction points to
// upper_ghost_data. Individual tensor components and different tensors are
// treated the same, so they all count in number_of_variables
template <typename DerivativeComputer, size_t Dim>
void second_logical_partial_derivatives_fastest_dim(
    const gsl::not_null<gsl::span<double>*> pure_second_derivative,
    const gsl::not_null<gsl::span<double>*> mixed_second_derivative,
    const gsl::span<const double>& volume_vars,
    const gsl::span<const double>& lower_ghost_data,
    const gsl::span<const double>& upper_ghost_data,
    const gsl::span<const double>& left_ghost_data,
    const gsl::span<const double>& right_ghost_data,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const double delta_squared) {
  constexpr size_t fd_order = DerivativeComputer::fd_order;
  ASSERT(Dim == 3, "The dimension is hardcoded to be 3 at this moment.");
  ASSERT(fd_order == 4,
         "Only the 4th order finite difference has been implemented");
  constexpr size_t ghost_zone_for_stencil =
      fd_order / 2;  // decide points of derivatives that need ghost data

  const std::array<double, 2> mixed_second_derivative_weights =
      DerivativeComputer::mixed_second_derivative_weights(1.0 / delta_squared);
  const std::array<double, 3> mixed_second_derivative_special_weights =
      DerivativeComputer::mixed_second_derivative_special_weights(
          1.0 / delta_squared);
  const std::array<double, 3> pure_second_derivative_weights =
      DerivativeComputer::pure_second_derivative_weights(1.0 / delta_squared);

  size_t number_of_stripes =
      volume_extents.slice_away(0).product() * number_of_variables;
  ASSERT(left_ghost_data.size() % number_of_stripes == 0,
         "The left ghost data must be a multiple of the number of stripes ("
             << number_of_stripes
             << "), which is defined as the number of variables ("
             << number_of_variables
             << ") times the number of grid points on a 2d slice ("
             << volume_extents.slice_away(0).product() << ")");
  ASSERT(volume_extents[0] >= 5 and volume_extents[1] >= 5 and
             volume_extents[2] >= 5,
         "At least 5 points must exist in each direction");
  ASSERT(right_ghost_data.size() == left_ghost_data.size(),
         "The left ghost data size ("
             << left_ghost_data.size()
             << ") must match the right ghost data size, "
             << right_ghost_data.size());
  const size_t ghost_pts_in_left_neighbor_data =
      left_ghost_data.size() / number_of_stripes;
  const Index<Dim> left_neighbor_extents(ghost_pts_in_left_neighbor_data,
                                         volume_extents[1], volume_extents[2]);

  // These following code can be simplified if we assume isotropic extents
  number_of_stripes =
      volume_extents.slice_away(1).product() * number_of_variables;
  ASSERT(lower_ghost_data.size() % number_of_stripes == 0,
         "The lower ghost data must be a multiple of the number of stripes ("
             << number_of_stripes
             << "), which is defined as the number of variables ("
             << number_of_variables
             << ") times the number of grid points on a 2d slice ("
             << volume_extents.slice_away(1).product() << ")");
  ASSERT(lower_ghost_data.size() == upper_ghost_data.size(),
         "The lower ghost data size ("
             << lower_ghost_data.size()
             << ") must match the upper ghost data size, "
             << upper_ghost_data.size());
  const size_t ghost_pts_in_lower_neighbor_data =
      lower_ghost_data.size() / number_of_stripes;
  const Index<Dim> lower_neighbor_extents(
      volume_extents[0], ghost_pts_in_lower_neighbor_data, volume_extents[2]);

  constexpr size_t special_mixed_stencil_width = 13;
  constexpr size_t mixed_stencil_width = 8;
  constexpr size_t pure_stencil_width = 5;
  std::array<double, special_mixed_stencil_width> q_mixed_special{};
  std::array<double, mixed_stencil_width> q_mixed{};
  std::array<double, pure_stencil_width> q_pure{};

  // deal with each variable or tensor component
  for (size_t vars_slice = 0; vars_slice < number_of_variables; ++vars_slice) {
    const size_t vars_slice_offset = vars_slice * volume_extents.product();
    const size_t left_neighbor_vars_slice_offset =
        vars_slice * left_ghost_data.size() / number_of_variables;
    const size_t lower_neighbor_vars_slice_offset =
        vars_slice * lower_ghost_data.size() / number_of_variables;

    // deal with each 2d xy slice
    for (size_t k = 0; k < volume_extents[2]; ++k) {
      // compute second derivatives in the bulk
      for (size_t i = ghost_zone_for_stencil;
           i < volume_extents[0] - ghost_zone_for_stencil; ++i) {
        for (size_t j = ghost_zone_for_stencil;
             j < volume_extents[1] - ghost_zone_for_stencil; ++j) {
          const size_t vars_collapsed_index =
              collapsed_index(Index<Dim>(i, j, k), volume_extents) +
              vars_slice_offset;

          q_pure[0] = volume_vars[vars_collapsed_index];
          q_pure[1] = volume_vars[vars_collapsed_index - 1];
          q_pure[2] = volume_vars[vars_collapsed_index + 1];
          q_pure[3] = volume_vars[vars_collapsed_index - 2];
          q_pure[4] = volume_vars[vars_collapsed_index + 2];

          (*pure_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::pure_second_deriv_pointwise(
                  q_pure, 1, pure_second_derivative_weights);

          q_mixed[0] =
              volume_vars[vars_collapsed_index + volume_extents[0] + 1];
          q_mixed[1] =
              volume_vars[vars_collapsed_index - volume_extents[0] - 1];
          q_mixed[2] =
              volume_vars[vars_collapsed_index - volume_extents[0] + 1];
          q_mixed[3] =
              volume_vars[vars_collapsed_index + volume_extents[0] - 1];
          q_mixed[4] =
              volume_vars[vars_collapsed_index + 2 * volume_extents[0] + 2];
          q_mixed[5] =
              volume_vars[vars_collapsed_index - 2 * volume_extents[0] - 2];
          q_mixed[6] =
              volume_vars[vars_collapsed_index - 2 * volume_extents[0] + 2];
          q_mixed[7] =
              volume_vars[vars_collapsed_index + 2 * volume_extents[0] - 2];

          (*mixed_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::mixed_second_deriv_pointwise(
                  q_mixed, 1, mixed_second_derivative_weights);
        }
      }

      // compute second derivatives in the ghost zone
      // compute second derivatives in the left and right strips
      const size_t right_ghost_zone_start =
          volume_extents[0] - ghost_zone_for_stencil;

      for (size_t j = ghost_zone_for_stencil;
           j < volume_extents[1] - ghost_zone_for_stencil; ++j) {
        // compute second derivatives in the left strip
        for (size_t i = 0; i < ghost_zone_for_stencil; ++i) {
          const size_t vars_collapsed_index =
              collapsed_index(Index<Dim>(i, j, k), volume_extents) +
              vars_slice_offset;

          q_pure[0] = volume_vars[vars_collapsed_index];
          const size_t neighbor_vars_collapsed_index =
              collapsed_index(
                  Index<Dim>(ghost_pts_in_left_neighbor_data - 1, j, k),
                  left_neighbor_extents) +
              left_neighbor_vars_slice_offset;
          q_pure[1] = (i == 0 ? left_ghost_data[neighbor_vars_collapsed_index]
                              : volume_vars[vars_collapsed_index - 1]);
          q_pure[2] = volume_vars[vars_collapsed_index + 1];
          q_pure[3] = left_ghost_data[neighbor_vars_collapsed_index - 1 + i];
          q_pure[4] = volume_vars[vars_collapsed_index + 2];

          (*pure_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::pure_second_deriv_pointwise(
                  q_pure, 1, pure_second_derivative_weights);

          q_mixed[0] =
              volume_vars[vars_collapsed_index + volume_extents[0] + 1];
          q_mixed[1] =
              (i == 0
                   ? left_ghost_data[neighbor_vars_collapsed_index -
                                     left_neighbor_extents[0]]
                   : volume_vars[vars_collapsed_index - volume_extents[0] - 1]);
          q_mixed[2] =
              volume_vars[vars_collapsed_index - volume_extents[0] + 1];
          q_mixed[3] =
              (i == 0
                   ? left_ghost_data[neighbor_vars_collapsed_index +
                                     left_neighbor_extents[0]]
                   : volume_vars[vars_collapsed_index + volume_extents[0] - 1]);
          q_mixed[4] =
              volume_vars[vars_collapsed_index + 2 * volume_extents[0] + 2];
          q_mixed[5] = left_ghost_data[neighbor_vars_collapsed_index -
                                       2 * left_neighbor_extents[0] - 1 + i];
          q_mixed[6] =
              volume_vars[vars_collapsed_index - 2 * volume_extents[0] + 2];
          q_mixed[7] = left_ghost_data[neighbor_vars_collapsed_index +
                                       2 * left_neighbor_extents[0] - 1 + i];

          (*mixed_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::mixed_second_deriv_pointwise(
                  q_mixed, 1, mixed_second_derivative_weights);
        }

        // compute second derivatives in the right strip
        for (size_t i = right_ghost_zone_start; i < volume_extents[0]; ++i) {
          const size_t vars_collapsed_index =
              collapsed_index(Index<Dim>(i, j, k), volume_extents) +
              vars_slice_offset;

          q_pure[0] = volume_vars[vars_collapsed_index];
          const size_t neighbor_vars_collapsed_index =
              collapsed_index(Index<Dim>(0, j, k), left_neighbor_extents) +
              left_neighbor_vars_slice_offset;
          q_pure[1] = (i == right_ghost_zone_start
                           ? volume_vars[vars_collapsed_index + 1]
                           : right_ghost_data[neighbor_vars_collapsed_index]);
          q_pure[2] = volume_vars[vars_collapsed_index - 1];
          q_pure[3] = right_ghost_data[neighbor_vars_collapsed_index + i -
                                       right_ghost_zone_start];
          q_pure[4] = volume_vars[vars_collapsed_index - 2];

          (*pure_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::pure_second_deriv_pointwise(
                  q_pure, 1, pure_second_derivative_weights);

          q_mixed[0] =
              (i == right_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + volume_extents[0] + 1]
                   : right_ghost_data[neighbor_vars_collapsed_index +
                                      left_neighbor_extents[0]]);
          q_mixed[1] =
              volume_vars[vars_collapsed_index - volume_extents[0] - 1];
          q_mixed[2] =
              (i == right_ghost_zone_start
                   ? volume_vars[vars_collapsed_index - volume_extents[0] + 1]
                   : right_ghost_data[neighbor_vars_collapsed_index -
                                      left_neighbor_extents[0]]);
          q_mixed[3] =
              volume_vars[vars_collapsed_index + volume_extents[0] - 1];
          q_mixed[4] = right_ghost_data[neighbor_vars_collapsed_index +
                                        2 * left_neighbor_extents[0] + i -
                                        right_ghost_zone_start];
          q_mixed[5] =
              volume_vars[vars_collapsed_index - 2 * volume_extents[0] - 2];
          q_mixed[6] = right_ghost_data[neighbor_vars_collapsed_index -
                                        2 * left_neighbor_extents[0] + i -
                                        right_ghost_zone_start];
          q_mixed[7] =
              volume_vars[vars_collapsed_index + 2 * volume_extents[0] - 2];

          (*mixed_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::mixed_second_deriv_pointwise(
                  q_mixed, 1, mixed_second_derivative_weights);
        }
      }

      const size_t upper_ghost_zone_start =
          volume_extents[1] - ghost_zone_for_stencil;

      // compute second derivatives in the upper and lower strips
      for (size_t i = ghost_zone_for_stencil;
           i < volume_extents[0] - ghost_zone_for_stencil; ++i) {
        // compute second derivatives in the lower strip
        for (size_t j = 0; j < ghost_zone_for_stencil; ++j) {
          const size_t vars_collapsed_index =
              collapsed_index(Index<Dim>(i, j, k), volume_extents) +
              vars_slice_offset;
          const size_t neighbor_vars_collapsed_index =
              collapsed_index(
                  Index<Dim>(i, ghost_pts_in_lower_neighbor_data - 1, k),
                  lower_neighbor_extents) +
              lower_neighbor_vars_slice_offset;

          q_pure[0] = volume_vars[vars_collapsed_index];
          q_pure[1] = volume_vars[vars_collapsed_index - 1];
          q_pure[2] = volume_vars[vars_collapsed_index + 1];
          q_pure[3] = volume_vars[vars_collapsed_index - 2];
          q_pure[4] = volume_vars[vars_collapsed_index + 2];

          (*pure_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::pure_second_deriv_pointwise(
                  q_pure, 1, pure_second_derivative_weights);

          q_mixed[0] =
              volume_vars[vars_collapsed_index + volume_extents[0] + 1];
          q_mixed[1] =
              (j == 0
                   ? lower_ghost_data[neighbor_vars_collapsed_index - 1]
                   : volume_vars[vars_collapsed_index - volume_extents[0] - 1]);
          q_mixed[2] =
              (j == 0
                   ? lower_ghost_data[neighbor_vars_collapsed_index + 1]
                   : volume_vars[vars_collapsed_index - volume_extents[0] + 1]);
          q_mixed[3] =
              volume_vars[vars_collapsed_index + volume_extents[0] - 1];
          q_mixed[4] =
              volume_vars[vars_collapsed_index + 2 * volume_extents[0] + 2];
          q_mixed[5] =
              lower_ghost_data[neighbor_vars_collapsed_index -
                               (1 - j) * lower_neighbor_extents[0] - 2];
          q_mixed[6] =
              lower_ghost_data[neighbor_vars_collapsed_index -
                               (1 - j) * lower_neighbor_extents[0] + 2];
          q_mixed[7] =
              volume_vars[vars_collapsed_index + 2 * volume_extents[0] - 2];

          (*mixed_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::mixed_second_deriv_pointwise(
                  q_mixed, 1, mixed_second_derivative_weights);
        }

        // compute second derivatives in the upper strip
        for (size_t j = upper_ghost_zone_start; j < volume_extents[1]; ++j) {
          const size_t vars_collapsed_index =
              collapsed_index(Index<Dim>(i, j, k), volume_extents) +
              vars_slice_offset;
          const size_t neighbor_vars_collapsed_index =
              collapsed_index(Index<Dim>(i, 0, k), lower_neighbor_extents) +
              lower_neighbor_vars_slice_offset;

          q_pure[0] = volume_vars[vars_collapsed_index];
          q_pure[1] = volume_vars[vars_collapsed_index - 1];
          q_pure[2] = volume_vars[vars_collapsed_index + 1];
          q_pure[3] = volume_vars[vars_collapsed_index - 2];
          q_pure[4] = volume_vars[vars_collapsed_index + 2];

          (*pure_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::pure_second_deriv_pointwise(
                  q_pure, 1, pure_second_derivative_weights);

          q_mixed[0] =
              (j == upper_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + volume_extents[0] + 1]
                   : upper_ghost_data[neighbor_vars_collapsed_index + 1]);
          q_mixed[1] =
              volume_vars[vars_collapsed_index - volume_extents[0] - 1];
          q_mixed[2] =
              volume_vars[vars_collapsed_index - volume_extents[0] + 1];
          q_mixed[3] =
              (j == upper_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + volume_extents[0] - 1]
                   : upper_ghost_data[neighbor_vars_collapsed_index - 1]);
          q_mixed[4] = upper_ghost_data[neighbor_vars_collapsed_index + 2 +
                                        (j - upper_ghost_zone_start) *
                                            lower_neighbor_extents[0]];
          q_mixed[5] =
              volume_vars[vars_collapsed_index - 2 * volume_extents[0] - 2];
          q_mixed[6] =
              volume_vars[vars_collapsed_index - 2 * volume_extents[0] + 2];
          q_mixed[7] = upper_ghost_data[neighbor_vars_collapsed_index - 2 +
                                        (j - upper_ghost_zone_start) *
                                            lower_neighbor_extents[0]];

          (*mixed_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::mixed_second_deriv_pointwise(
                  q_mixed, 1, mixed_second_derivative_weights);
        }
      }

      // compute second derivatives in the lower left corner
      for (size_t i = 0; i < ghost_zone_for_stencil; ++i) {
        for (size_t j = 0; j < ghost_zone_for_stencil; ++j) {
          const size_t vars_collapsed_index =
              collapsed_index(Index<Dim>(i, j, k), volume_extents) +
              vars_slice_offset;
          const size_t left_neighbor_vars_collapsed_index =
              collapsed_index(
                  Index<Dim>(ghost_pts_in_left_neighbor_data - 1, j, k),
                  left_neighbor_extents) +
              left_neighbor_vars_slice_offset;
          const size_t lower_neighbor_vars_collapsed_index =
              collapsed_index(
                  Index<Dim>(i, ghost_pts_in_lower_neighbor_data - 1, k),
                  lower_neighbor_extents) +
              lower_neighbor_vars_slice_offset;

          q_pure[0] = volume_vars[vars_collapsed_index];
          q_pure[1] =
              (i == 0 ? left_ghost_data[left_neighbor_vars_collapsed_index]
                      : volume_vars[vars_collapsed_index - 1]);
          q_pure[2] = volume_vars[vars_collapsed_index + 1];
          q_pure[3] =
              left_ghost_data[left_neighbor_vars_collapsed_index - 1 + i];
          q_pure[4] = volume_vars[vars_collapsed_index + 2];

          (*pure_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::pure_second_deriv_pointwise(
                  q_pure, 1, pure_second_derivative_weights);

          q_mixed_special[0] = volume_vars[vars_collapsed_index];
          q_mixed_special[1] =
              (i == 0 ? left_ghost_data[left_neighbor_vars_collapsed_index]
                      : volume_vars[vars_collapsed_index - 1]);
          q_mixed_special[2] = volume_vars[vars_collapsed_index + 1];
          q_mixed_special[3] =
              (j == 0 ? lower_ghost_data[lower_neighbor_vars_collapsed_index]
                      : volume_vars[vars_collapsed_index - volume_extents[0]]);
          q_mixed_special[4] =
              volume_vars[vars_collapsed_index + volume_extents[0]];
          q_mixed_special[5] =
              (i == 0
                   ? left_ghost_data[left_neighbor_vars_collapsed_index +
                                     left_neighbor_extents[0]]
                   : volume_vars[vars_collapsed_index + volume_extents[0] - 1]);
          q_mixed_special[6] =
              (j == 0
                   ? lower_ghost_data[lower_neighbor_vars_collapsed_index + 1]
                   : volume_vars[vars_collapsed_index - volume_extents[0] + 1]);
          q_mixed_special[7] =
              left_ghost_data[left_neighbor_vars_collapsed_index - 1 + i];
          q_mixed_special[8] = volume_vars[vars_collapsed_index + 2];
          q_mixed_special[9] =
              lower_ghost_data[lower_neighbor_vars_collapsed_index -
                               (1 - j) * lower_neighbor_extents[0]];
          q_mixed_special[10] =
              volume_vars[vars_collapsed_index + 2 * volume_extents[0]];
          q_mixed_special[11] =
              left_ghost_data[left_neighbor_vars_collapsed_index +
                              2 * left_neighbor_extents[0] - 1 + i];
          q_mixed_special[12] =
              lower_ghost_data[lower_neighbor_vars_collapsed_index -
                               (1 - j) * lower_neighbor_extents[0] + 2];

          (*mixed_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::mixed_second_deriv_special_pointwise(
                  q_mixed_special, 1, mixed_second_derivative_special_weights);
        }
      }

      // compute second derivative in the upper left corner
      for (size_t i = 0; i < ghost_zone_for_stencil; ++i) {
        for (size_t j = upper_ghost_zone_start; j < volume_extents[1]; ++j) {
          const size_t vars_collapsed_index =
              collapsed_index(Index<Dim>(i, j, k), volume_extents) +
              vars_slice_offset;
          const size_t left_neighbor_vars_collapsed_index =
              collapsed_index(
                  Index<Dim>(ghost_pts_in_left_neighbor_data - 1, j, k),
                  left_neighbor_extents) +
              left_neighbor_vars_slice_offset;
          const size_t upper_neighbor_vars_collapsed_index =
              collapsed_index(Index<Dim>(i, 0, k), lower_neighbor_extents) +
              lower_neighbor_vars_slice_offset;

          q_pure[0] = volume_vars[vars_collapsed_index];
          q_pure[1] =
              (i == 0 ? left_ghost_data[left_neighbor_vars_collapsed_index]
                      : volume_vars[vars_collapsed_index - 1]);
          q_pure[2] = volume_vars[vars_collapsed_index + 1];
          q_pure[3] =
              left_ghost_data[left_neighbor_vars_collapsed_index - 1 + i];
          q_pure[4] = volume_vars[vars_collapsed_index + 2];

          (*pure_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::pure_second_deriv_pointwise(
                  q_pure, 1, pure_second_derivative_weights);

          q_mixed_special[0] = volume_vars[vars_collapsed_index];
          q_mixed_special[1] =
              (i == 0 ? left_ghost_data[left_neighbor_vars_collapsed_index]
                      : volume_vars[vars_collapsed_index - 1]);
          q_mixed_special[2] = volume_vars[vars_collapsed_index + 1];
          q_mixed_special[3] =
              volume_vars[vars_collapsed_index - volume_extents[0]];
          q_mixed_special[4] =
              (j == upper_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + volume_extents[0]]
                   : upper_ghost_data[upper_neighbor_vars_collapsed_index]);
          q_mixed_special[5] =
              (i == 0
                   ? left_ghost_data[left_neighbor_vars_collapsed_index -
                                     left_neighbor_extents[0]]
                   : volume_vars[vars_collapsed_index - volume_extents[0] - 1]);
          q_mixed_special[6] =
              (j == upper_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + volume_extents[0] + 1]
                   : upper_ghost_data[upper_neighbor_vars_collapsed_index + 1]);
          q_mixed_special[7] =
              left_ghost_data[left_neighbor_vars_collapsed_index - 1 + i];
          q_mixed_special[8] = volume_vars[vars_collapsed_index + 2];
          q_mixed_special[9] =
              volume_vars[vars_collapsed_index - 2 * volume_extents[0]];
          q_mixed_special[10] =
              upper_ghost_data[upper_neighbor_vars_collapsed_index +
                               (j - upper_ghost_zone_start) *
                                   lower_neighbor_extents[0]];
          q_mixed_special[11] =
              left_ghost_data[left_neighbor_vars_collapsed_index -
                              2 * left_neighbor_extents[0] - 1 + i];
          q_mixed_special[12] =
              upper_ghost_data[upper_neighbor_vars_collapsed_index + 2 +
                               (j - upper_ghost_zone_start) *
                                   lower_neighbor_extents[0]];

          // -1.0 as the weights are opposite for asending diagonal of the
          // stencil
          (*mixed_second_derivative)[vars_collapsed_index] =
              -1.0 *
              DerivativeComputer::mixed_second_deriv_special_pointwise(
                  q_mixed_special, 1, mixed_second_derivative_special_weights);
        }
      }

      // compute second derivatives in the lower right corner
      for (size_t i = right_ghost_zone_start; i < volume_extents[0]; ++i) {
        for (size_t j = 0; j < ghost_zone_for_stencil; ++j) {
          const size_t vars_collapsed_index =
              collapsed_index(Index<Dim>(i, j, k), volume_extents) +
              vars_slice_offset;
          const size_t right_neighbor_vars_collapsed_index =
              collapsed_index(Index<Dim>(0, j, k), left_neighbor_extents) +
              left_neighbor_vars_slice_offset;
          const size_t lower_neighbor_vars_collapsed_index =
              collapsed_index(
                  Index<Dim>(i, ghost_pts_in_lower_neighbor_data - 1, k),
                  lower_neighbor_extents) +
              lower_neighbor_vars_slice_offset;

          q_pure[0] = volume_vars[vars_collapsed_index];
          q_pure[1] =
              (i == right_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + 1]
                   : right_ghost_data[right_neighbor_vars_collapsed_index]);
          q_pure[2] = volume_vars[vars_collapsed_index - 1];
          q_pure[3] = right_ghost_data[right_neighbor_vars_collapsed_index + i -
                                       right_ghost_zone_start];
          q_pure[4] = volume_vars[vars_collapsed_index - 2];

          (*pure_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::pure_second_deriv_pointwise(
                  q_pure, 1, pure_second_derivative_weights);

          q_mixed_special[0] = volume_vars[vars_collapsed_index];
          q_mixed_special[1] = volume_vars[vars_collapsed_index - 1];
          q_mixed_special[2] =
              (i == right_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + 1]
                   : right_ghost_data[right_neighbor_vars_collapsed_index]);
          q_mixed_special[3] =
              (j == 0 ? lower_ghost_data[lower_neighbor_vars_collapsed_index]
                      : volume_vars[vars_collapsed_index - volume_extents[0]]);
          q_mixed_special[4] =
              volume_vars[vars_collapsed_index + volume_extents[0]];
          q_mixed_special[5] =
              (j == 0
                   ? lower_ghost_data[lower_neighbor_vars_collapsed_index - 1]
                   : volume_vars[vars_collapsed_index - volume_extents[0] - 1]);
          q_mixed_special[6] =
              (i == right_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + volume_extents[0] + 1]
                   : right_ghost_data[right_neighbor_vars_collapsed_index +
                                      left_neighbor_extents[0]]);
          q_mixed_special[7] = volume_vars[vars_collapsed_index - 2];
          q_mixed_special[8] =
              right_ghost_data[right_neighbor_vars_collapsed_index + i -
                               right_ghost_zone_start];
          q_mixed_special[9] =
              lower_ghost_data[lower_neighbor_vars_collapsed_index -
                               (1 - j) * lower_neighbor_extents[0]];
          q_mixed_special[10] =
              volume_vars[vars_collapsed_index + 2 * volume_extents[0]];
          q_mixed_special[11] =
              lower_ghost_data[lower_neighbor_vars_collapsed_index -
                               (1 - j) * lower_neighbor_extents[0] - 2];
          q_mixed_special[12] =
              right_ghost_data[right_neighbor_vars_collapsed_index +
                               2 * left_neighbor_extents[0] + i -
                               right_ghost_zone_start];

          // -1.0 as the weights are opposite for asending diagonal of the
          // stencil
          (*mixed_second_derivative)[vars_collapsed_index] =
              -1.0 *
              DerivativeComputer::mixed_second_deriv_special_pointwise(
                  q_mixed_special, 1, mixed_second_derivative_special_weights);
        }
      }

      // compute second derivatives in the upper right corner
      for (size_t i = right_ghost_zone_start; i < volume_extents[0]; ++i) {
        for (size_t j = upper_ghost_zone_start; j < volume_extents[1]; ++j) {
          const size_t vars_collapsed_index =
              collapsed_index(Index<Dim>(i, j, k), volume_extents) +
              vars_slice_offset;
          const size_t right_neighbor_vars_collapsed_index =
              collapsed_index(Index<Dim>(0, j, k), left_neighbor_extents) +
              left_neighbor_vars_slice_offset;
          const size_t upper_neighbor_vars_collapsed_index =
              collapsed_index(Index<Dim>(i, 0, k), lower_neighbor_extents) +
              lower_neighbor_vars_slice_offset;

          q_pure[0] = volume_vars[vars_collapsed_index];
          q_pure[1] =
              (i == right_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + 1]
                   : right_ghost_data[right_neighbor_vars_collapsed_index]);
          q_pure[2] = volume_vars[vars_collapsed_index - 1];
          q_pure[3] = right_ghost_data[right_neighbor_vars_collapsed_index + i -
                                       right_ghost_zone_start];
          q_pure[4] = volume_vars[vars_collapsed_index - 2];

          (*pure_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::pure_second_deriv_pointwise(
                  q_pure, 1, pure_second_derivative_weights);

          q_mixed_special[0] = volume_vars[vars_collapsed_index];
          q_mixed_special[1] = volume_vars[vars_collapsed_index - 1];
          q_mixed_special[2] =
              (i == right_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + 1]
                   : right_ghost_data[right_neighbor_vars_collapsed_index]);
          q_mixed_special[3] =
              volume_vars[vars_collapsed_index - volume_extents[0]];
          q_mixed_special[4] =
              (j == upper_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + volume_extents[0]]
                   : upper_ghost_data[upper_neighbor_vars_collapsed_index]);
          q_mixed_special[5] =
              (j == upper_ghost_zone_start
                   ? volume_vars[vars_collapsed_index + volume_extents[0] - 1]
                   : upper_ghost_data[upper_neighbor_vars_collapsed_index - 1]);
          q_mixed_special[6] =
              (i == right_ghost_zone_start
                   ? volume_vars[vars_collapsed_index - volume_extents[0] + 1]
                   : right_ghost_data[right_neighbor_vars_collapsed_index -
                                      left_neighbor_extents[0]]);
          q_mixed_special[7] = volume_vars[vars_collapsed_index - 2];
          q_mixed_special[8] =
              right_ghost_data[right_neighbor_vars_collapsed_index + i -
                               right_ghost_zone_start];
          q_mixed_special[9] =
              volume_vars[vars_collapsed_index - 2 * volume_extents[0]];
          q_mixed_special[10] =
              upper_ghost_data[upper_neighbor_vars_collapsed_index +
                               (j - upper_ghost_zone_start) *
                                   lower_neighbor_extents[0]];
          q_mixed_special[11] =
              upper_ghost_data[upper_neighbor_vars_collapsed_index - 2 +
                               (j - upper_ghost_zone_start) *
                                   lower_neighbor_extents[0]];
          q_mixed_special[12] =
              right_ghost_data[right_neighbor_vars_collapsed_index -
                               2 * left_neighbor_extents[0] + i -
                               right_ghost_zone_start];

          (*mixed_second_derivative)[vars_collapsed_index] =
              DerivativeComputer::mixed_second_deriv_special_pointwise(
                  q_mixed_special, 1, mixed_second_derivative_special_weights);
        }
      }
    }
  }
}

template <typename DerivativeComputer, size_t Dim>
void second_logical_partial_derivatives_impl(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        pure_second_logical_derivatives,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        mixed_second_logical_derivatives,
    gsl::span<double>* const in_buffer,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables) {
  const size_t number_of_points = volume_mesh.number_of_grid_points();
#ifdef SPECTRE_DEBUG
  ASSERT(Dim == 3, "The dimension is hardcoded to be 3 at this moment.");
  ASSERT(volume_mesh == Mesh<Dim>(volume_mesh.extents(0), volume_mesh.basis(0),
                                  volume_mesh.quadrature(0)),
         "The mesh must be isotropic, but got " << volume_mesh);
  ASSERT(
      volume_mesh.basis(0) == Spectral::Basis::FiniteDifference,
      "Mesh basis must be FiniteDifference but got " << volume_mesh.basis(0));
  ASSERT(volume_mesh.quadrature(0) == Spectral::Quadrature::CellCentered,
         "Mesh quadrature must be CellCentered but got "
             << volume_mesh.quadrature(0));
  // Note that number_of_variables here refers to number of independent
  // components; e.g. a rank-1 tensor has 3 independent components
  ASSERT(volume_vars.size() == number_of_points * number_of_variables,
         "The size of the volume vars must be the number of points ("
             << number_of_points << ") times the number of variables ("
             << number_of_variables << ") but is " << volume_vars.size());
  for (size_t i = 0; i < Dim; ++i) {
    ASSERT(gsl::at(*pure_second_logical_derivatives, i).size() ==
               volume_vars.size(),
           "The pure second logical derivatives must have size "
               << volume_vars.size() << " but has size "
               << gsl::at(*pure_second_logical_derivatives, i).size()
               << " in dimension " << i);
    ASSERT(gsl::at(*mixed_second_logical_derivatives, i).size() ==
               volume_vars.size(),
           "The mixed second logical derivatives must have size "
               << volume_vars.size() << " but has size "
               << gsl::at(*mixed_second_logical_derivatives, i).size()
               << " in dimension " << i);
  }
#endif  // SPECTRE_DEBUG

  ASSERT(ghost_cell_vars.contains(Direction<Dim>::lower_xi()),
         "Couldn't find ghost data in lower-xi");
  ASSERT(ghost_cell_vars.contains(Direction<Dim>::upper_xi()),
         "Couldn't find ghost data in upper-xi");
  ASSERT(ghost_cell_vars.contains(Direction<Dim>::lower_eta()),
         "Couldn't find ghost data in lower-eta");
  ASSERT(ghost_cell_vars.contains(Direction<Dim>::upper_eta()),
         "Couldn't find ghost data in upper-eta");
  ASSERT(ghost_cell_vars.contains(Direction<Dim>::lower_zeta()),
         "Couldn't find ghost data in lower-zeta");
  ASSERT(ghost_cell_vars.contains(Direction<Dim>::upper_zeta()),
         "Couldn't find ghost data in upper-zeta");

  const Index<Dim>& volume_extents = volume_mesh.extents();

  const auto& logical_xi_coords =
      Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                   Spectral::Quadrature::CellCentered>(
          volume_mesh.extents(0));

  // compute the xx, xy derivatives
  second_logical_partial_derivatives_fastest_dim<DerivativeComputer>(
      make_not_null(&(*pure_second_logical_derivatives)[0]),
      make_not_null(&(*mixed_second_logical_derivatives)[0]), volume_vars,
      ghost_cell_vars.at(Direction<Dim>::lower_eta()),  // lower ghost data
      ghost_cell_vars.at(Direction<Dim>::upper_eta()),  // upper ghost data
      ghost_cell_vars.at(Direction<Dim>::lower_xi()),   // left ghost data
      ghost_cell_vars.at(Direction<Dim>::upper_xi()),   // right ghost data
      volume_extents, number_of_variables,
      square(logical_xi_coords[1] - logical_xi_coords[0]));

  // We transpose from (x,y,z,vars) ordering to (y,z,x,vars) ordering
  // Might not be the most efficient (unclear), but easiest.
  // We use a single large buffer for both the yz and xz derivatives
  // to reduce the number of memory allocations and improve data locality.

  // compute the yy, yz derivatives
  auto& left_ghost = ghost_cell_vars.at(Direction<Dim>::lower_eta());
  auto& right_ghost = ghost_cell_vars.at(Direction<Dim>::upper_eta());
  auto& lower_ghost = ghost_cell_vars.at(Direction<Dim>::lower_zeta());
  auto& upper_ghost = ghost_cell_vars.at(Direction<Dim>::upper_zeta());
  const size_t derivative_size = (*pure_second_logical_derivatives)[1].size() +
                                 (*mixed_second_logical_derivatives)[1].size();
  DataVector buffer{};
  if (in_buffer != nullptr) {
    ASSERT(
        (in_buffer->size() >= volume_vars.size() + lower_ghost.size() +
                                  upper_ghost.size() + left_ghost.size() +
                                  right_ghost.size() + derivative_size),
        "The buffer must have size greater than or equal to "
            << (volume_vars.size() + lower_ghost.size() + upper_ghost.size() +
                left_ghost.size() + right_ghost.size() + derivative_size)
            << " but has size " << in_buffer->size());
    buffer.set_data_ref(in_buffer->data(), in_buffer->size());
  } else {
    buffer = DataVector{volume_vars.size() + lower_ghost.size() +
                        upper_ghost.size() + left_ghost.size() +
                        right_ghost.size() + derivative_size};
  }

  size_t number_of_pts_in_left_ghost = left_ghost.size() / number_of_variables;
  size_t number_of_pts_in_right_ghost =
      right_ghost.size() / number_of_variables;
  size_t number_of_pts_in_lower_ghost =
      lower_ghost.size() / number_of_variables;
  size_t number_of_pts_in_upper_ghost =
      upper_ghost.size() / number_of_variables;

  for (size_t vars_slice = 0; vars_slice < number_of_variables; ++vars_slice) {
    raw_transpose(
        make_not_null(&buffer[vars_slice * number_of_points]),
        volume_vars.subspan(vars_slice * number_of_points, number_of_points)
            .data(),
        volume_extents[0], number_of_points / volume_extents[0]);
    raw_transpose(
        make_not_null(&buffer[volume_vars.size() +
                              vars_slice * number_of_pts_in_lower_ghost]),
        lower_ghost
            .subspan(vars_slice * number_of_pts_in_lower_ghost,
                     number_of_pts_in_lower_ghost)
            .data(),
        volume_extents[0], number_of_pts_in_lower_ghost / volume_extents[0]);
    raw_transpose(
        make_not_null(&buffer[volume_vars.size() + lower_ghost.size() +
                              vars_slice * number_of_pts_in_upper_ghost]),
        upper_ghost
            .subspan(vars_slice * number_of_pts_in_upper_ghost,
                     number_of_pts_in_upper_ghost)
            .data(),
        volume_extents[0], number_of_pts_in_upper_ghost / volume_extents[0]);

    raw_transpose(
        make_not_null(&buffer[volume_vars.size() + lower_ghost.size() +
                              upper_ghost.size() +
                              vars_slice * number_of_pts_in_left_ghost]),
        left_ghost
            .subspan(vars_slice * number_of_pts_in_left_ghost,
                     number_of_pts_in_left_ghost)
            .data(),
        volume_extents[0], number_of_pts_in_left_ghost / volume_extents[0]);

    raw_transpose(
        make_not_null(&buffer[volume_vars.size() + lower_ghost.size() +
                              upper_ghost.size() + left_ghost.size() +
                              vars_slice * number_of_pts_in_right_ghost]),
        right_ghost
            .subspan(vars_slice * number_of_pts_in_right_ghost,
                     number_of_pts_in_right_ghost)
            .data(),
        volume_extents[0], number_of_pts_in_right_ghost / volume_extents[0]);
  }

  // Note: assumes isotropic extents
  const size_t pure_second_derivative_offset_in_buffer =
      volume_vars.size() + lower_ghost.size() + upper_ghost.size() +
      left_ghost.size() + right_ghost.size();
  const size_t mixed_second_derivative_offset_in_buffer =
      pure_second_derivative_offset_in_buffer +
      (*pure_second_logical_derivatives)[1].size();
  gsl::span<double> pure_second_derivative_view =
      gsl::make_span(&buffer[pure_second_derivative_offset_in_buffer],
                     (*pure_second_logical_derivatives)[1].size());
  gsl::span<double> mixed_second_derivative_view =
      gsl::make_span(&buffer[mixed_second_derivative_offset_in_buffer],
                     (*mixed_second_logical_derivatives)[1].size());

  const auto& logical_eta_coords =
      Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                   Spectral::Quadrature::CellCentered>(
          volume_mesh.extents(1));

  second_logical_partial_derivatives_fastest_dim<DerivativeComputer>(
      make_not_null(&pure_second_derivative_view),
      make_not_null(&mixed_second_derivative_view),
      // NOLINTNEXTLINE(readability-container-data-pointer)
      gsl::make_span(&buffer[0], volume_vars.size()),
      gsl::make_span(&buffer[volume_vars.size()], lower_ghost.size()),
      gsl::make_span(&buffer[volume_vars.size() + lower_ghost.size()],
                     upper_ghost.size()),
      gsl::make_span(
          &buffer[volume_vars.size() + lower_ghost.size() + upper_ghost.size()],
          left_ghost.size()),
      gsl::make_span(&buffer[volume_vars.size() + lower_ghost.size() +
                             upper_ghost.size() + left_ghost.size()],
                     right_ghost.size()),
      volume_extents, number_of_variables,
      square(logical_eta_coords[1] - logical_eta_coords[0]));

  // Transpose result back
  for (size_t vars_slice = 0; vars_slice < number_of_variables; ++vars_slice) {
    raw_transpose(make_not_null((*pure_second_logical_derivatives)[1].data() +
                                vars_slice * number_of_points),
                  pure_second_derivative_view
                      .subspan(vars_slice * number_of_points, number_of_points)
                      .data(),
                  number_of_points / volume_extents[0], volume_extents[0]);

    raw_transpose(make_not_null((*mixed_second_logical_derivatives)[1].data() +
                                vars_slice * number_of_points),
                  mixed_second_derivative_view
                      .subspan(vars_slice * number_of_points, number_of_points)
                      .data(),
                  number_of_points / volume_extents[0], volume_extents[0]);
  }

  // compute the zz, zx derivatives
  number_of_pts_in_left_ghost =
      ghost_cell_vars.at(Direction<Dim>::lower_zeta()).size() /
      number_of_variables;
  number_of_pts_in_right_ghost =
      ghost_cell_vars.at(Direction<Dim>::upper_zeta()).size() /
      number_of_variables;
  number_of_pts_in_lower_ghost =
      ghost_cell_vars.at(Direction<Dim>::lower_xi()).size() /
      number_of_variables;
  number_of_pts_in_upper_ghost =
      ghost_cell_vars.at(Direction<Dim>::upper_xi()).size() /
      number_of_variables;

  const size_t chunk_size = volume_extents[0] * volume_extents[1];
  const size_t lower_chunk_size =
      number_of_pts_in_lower_ghost / volume_extents[2];
  const size_t upper_chunk_size =
      number_of_pts_in_upper_ghost / volume_extents[2];
  const size_t number_of_volume_chunks = number_of_points / chunk_size;
  const size_t number_of_left_neighbor_chunks =
      number_of_pts_in_left_ghost / chunk_size;
  const size_t number_of_right_neighbor_chunks =
      number_of_pts_in_right_ghost / chunk_size;
  const size_t number_of_lower_neighbor_chunks =
      number_of_pts_in_lower_ghost / lower_chunk_size;
  const size_t number_of_upper_neighbor_chunks =
      number_of_pts_in_upper_ghost / upper_chunk_size;

  for (size_t vars_slice = 0; vars_slice < number_of_variables; ++vars_slice) {
    raw_transpose(
        make_not_null(buffer.data() + vars_slice * number_of_points),
        volume_vars.subspan(vars_slice * number_of_points, number_of_points)
            .data(),
        chunk_size, number_of_volume_chunks);
    raw_transpose(
        make_not_null(&buffer[volume_vars.size() +
                              vars_slice * number_of_pts_in_lower_ghost]),
        ghost_cell_vars.at(Direction<Dim>::lower_xi())
            .subspan(vars_slice * number_of_pts_in_lower_ghost,
                     number_of_pts_in_lower_ghost)
            .data(),
        lower_chunk_size, number_of_lower_neighbor_chunks);
    raw_transpose(
        make_not_null(
            &buffer[volume_vars.size() +
                    ghost_cell_vars.at(Direction<Dim>::lower_xi()).size() +
                    vars_slice * number_of_pts_in_upper_ghost]),
        ghost_cell_vars.at(Direction<Dim>::upper_xi())
            .subspan(vars_slice * number_of_pts_in_upper_ghost,
                     number_of_pts_in_upper_ghost)
            .data(),
        upper_chunk_size, number_of_upper_neighbor_chunks);
    raw_transpose(
        make_not_null(
            &buffer[volume_vars.size() +
                    ghost_cell_vars.at(Direction<Dim>::lower_xi()).size() +
                    ghost_cell_vars.at(Direction<Dim>::upper_xi()).size() +
                    vars_slice * number_of_pts_in_left_ghost]),
        ghost_cell_vars.at(Direction<Dim>::lower_zeta())
            .subspan(vars_slice * number_of_pts_in_left_ghost,
                     number_of_pts_in_left_ghost)
            .data(),
        chunk_size, number_of_left_neighbor_chunks);
    raw_transpose(
        make_not_null(
            &buffer[volume_vars.size() +
                    ghost_cell_vars.at(Direction<Dim>::lower_xi()).size() +
                    ghost_cell_vars.at(Direction<Dim>::upper_xi()).size() +
                    ghost_cell_vars.at(Direction<Dim>::lower_zeta()).size() +
                    vars_slice * number_of_pts_in_right_ghost]),
        ghost_cell_vars.at(Direction<Dim>::upper_zeta())
            .subspan(vars_slice * number_of_pts_in_right_ghost,
                     number_of_pts_in_right_ghost)
            .data(),
        chunk_size, number_of_right_neighbor_chunks);
  }

  const auto& logical_zeta_coords =
      Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                   Spectral::Quadrature::CellCentered>(
          volume_mesh.extents(2));

  second_logical_partial_derivatives_fastest_dim<DerivativeComputer>(
      make_not_null(&pure_second_derivative_view),
      make_not_null(&mixed_second_derivative_view),
      // NOLINTNEXTLINE(readability-container-data-pointer)
      gsl::make_span(&buffer[0], volume_vars.size()),
      gsl::make_span(&buffer[volume_vars.size()],
                     ghost_cell_vars.at(Direction<Dim>::lower_xi()).size()),
      gsl::make_span(
          &buffer[volume_vars.size() +
                  ghost_cell_vars.at(Direction<Dim>::lower_xi()).size()],
          ghost_cell_vars.at(Direction<Dim>::upper_xi()).size()),
      gsl::make_span(
          &buffer[volume_vars.size() +
                  ghost_cell_vars.at(Direction<Dim>::lower_xi()).size() +
                  ghost_cell_vars.at(Direction<Dim>::upper_xi()).size()],
          ghost_cell_vars.at(Direction<Dim>::lower_zeta()).size()),
      gsl::make_span(
          &buffer[volume_vars.size() +
                  ghost_cell_vars.at(Direction<Dim>::lower_xi()).size() +
                  ghost_cell_vars.at(Direction<Dim>::upper_xi()).size() +
                  ghost_cell_vars.at(Direction<Dim>::lower_zeta()).size()],
          ghost_cell_vars.at(Direction<Dim>::upper_zeta()).size()),
      volume_extents, number_of_variables,
      square(logical_zeta_coords[1] - logical_zeta_coords[0]));

  // Transpose result back
  for (size_t vars_slice = 0; vars_slice < number_of_variables; ++vars_slice) {
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    raw_transpose(make_not_null((*pure_second_logical_derivatives)[2].data() +
                                vars_slice * number_of_points),
                  pure_second_derivative_view
                      .subspan(vars_slice * number_of_points, number_of_points)
                      .data(),
                  number_of_volume_chunks, chunk_size);
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    raw_transpose(make_not_null((*mixed_second_logical_derivatives)[2].data() +
                                vars_slice * number_of_points),
                  mixed_second_derivative_view
                      .subspan(vars_slice * number_of_points, number_of_points)
                      .data(),
                  number_of_volume_chunks, chunk_size);
  }
}
}  // namespace

namespace detail {
template <size_t Dim>
void second_logical_partial_derivatives_impl(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        pure_second_logical_derivatives,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        mixed_second_logical_derivatives,
    gsl::span<double>* const buffer, const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order) {
  switch (fd_order) {
    case 4:
      ::fd::second_logical_partial_derivatives_impl<ComputeImpl<4, true>>(
          pure_second_logical_derivatives, mixed_second_logical_derivatives,
          buffer, volume_vars, ghost_cell_vars, volume_mesh,
          number_of_variables);
      break;
    default:
      ERROR("Cannot do finite difference derivative of order " << fd_order);
  };
}
}  // namespace detail

template <size_t Dim>
void second_logical_partial_derivatives(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        pure_second_logical_derivatives,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        mixed_second_logical_derivatives,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order) {
  detail::second_logical_partial_derivatives_impl(
      pure_second_logical_derivatives, mixed_second_logical_derivatives,
      nullptr, volume_vars, ghost_cell_vars, volume_mesh, number_of_variables,
      fd_order);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template void detail::second_logical_partial_derivatives_impl(               \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          pure_second_derivative,                                              \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          mixed_second_derivative,                                             \
      gsl::span<double>* const buffer,                                         \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Mesh<DIM(data)>& volume_fd_mesh, size_t number_of_variables,       \
      size_t fd_order);                                                        \
  template void second_logical_partial_derivatives(                            \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          pure_second_derivative,                                              \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          mixed_second_derivative,                                             \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Mesh<DIM(data)>& volume_fd_mesh, size_t number_of_variables,       \
      size_t fd_order);

GENERATE_INSTANTIATIONS(INSTANTIATION, (3))

#undef GET_DIM
#undef INSTANTIATION

}  // namespace fd
