// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

#include "DataStructures/Index.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {

/*!
 * \brief Performs linear interpolation in arbitrary dimensions.
 *  The class is non-owning and expects a C-ordered array, (n, x, y, z).
 *  The variable index, n, varies fastest in memory.
 *  Note that this class is intentionally non-pupable.
 *
 *  \tparam Dimension dimensionality of the table
 *  \tparam NumberOfVariables number of variables stored in the table
 *  \tparam UniformSpacing indicated whether the table has uniform _spacing or
 * not. This is useful for performance reasons in finding the appropriate table
 * index.
 */

template <size_t Dimension, size_t NumberOfVariables, bool UniformSpacing>
class MultiLinearSpanInterpolation {

 public:
  template <size_t ThisDimension>
  struct Weight {
    std::array<double, two_to_the(ThisDimension)> weights;
    Index<two_to_the(ThisDimension)> index;
  };

  size_t extents(const size_t which_dimension) const {
    return number_of_points_[which_dimension];
  }

  void extrapolate_above_data(const size_t which_dimension, const bool value) {
    allow_extrapolation_abov_data_[which_dimension] = value;
  }

  void extrapolate_below_data(const size_t which_dimension, const bool value) {
    allow_extrapolation_below_data_[which_dimension] = value;
  }

  /// Compute interpolation weights for 1D tables
  Weight<1> get_weights(const double x1) const;

  /// Compute interpolation weights for 2D tables
  Weight<2> get_weights(const double x1, const double x2) const;

  /// Compute interpolation weights for 3D tables
  Weight<3> get_weights(const double x1, const double x2,
                        const double x3) const;

  double interpolate(const Weight<Dimension>& weights,
                     const size_t which_variable = 0) const {
    double result = 0;

    for (size_t nn = 0; nn < weights.weights.size(); ++nn) {
      result += weights.weights[nn] *
                y_[which_variable + NumberOfVariables * weights.index[nn]];
    }

    return result;
  }

  template <size_t... variables_to_interpolate>
  std::array<double, sizeof...(variables_to_interpolate)> interpolate(
      const Weight<Dimension>& weights) const {
    static_assert(sizeof...(variables_to_interpolate) <= NumberOfVariables,
                  "You are trying to interpolate more variables than this "
                  "container holds.");

    return std::array<double, sizeof...(variables_to_interpolate)>{
        interpolate(weights, variables_to_interpolate)...};
  }

  template <size_t NumberOfVariablesToInterpolate, typename... T,
            Requires<(std::is_floating_point_v<typename std::remove_cv_t<T>> and
                      ...)> = nullptr>
  std::array<double, NumberOfVariablesToInterpolate> interpolate(
      std::array<size_t, NumberOfVariablesToInterpolate>&
          variables_to_interpolate,
      const T&... target_points) const {
    static_assert(
        sizeof...(T) == Dimension,
        "You need to provide the correct number of interpolation points");
    static_assert(NumberOfVariablesToInterpolate <= NumberOfVariables,
                  "You are trying to interpolate more variables than this "
                  "container holds.");

    auto weights = get_weights(target_points...);

    std::array<double, NumberOfVariablesToInterpolate> interpolatedValues;

    for (size_t nn = 0; nn < NumberOfVariablesToInterpolate; ++nn) {
      interpolatedValues[nn] =
          interpolate(weights, variables_to_interpolate[nn]);
    }

    return interpolatedValues;
  }

  template <size_t NumberOfVariablesToInterpolate, size_t... I>
  std::array<double, NumberOfVariablesToInterpolate> interpolate(
      std::array<size_t, NumberOfVariablesToInterpolate>&
          variables_to_interpolate,
      std::array<double, Dimension>& target_points,
      std::index_sequence<I...>) const {
    return interpolate(variables_to_interpolate, target_points[I]...);
  }

  template <size_t NumberOfVariablesToInterpolate>
  std::array<double, NumberOfVariablesToInterpolate> interpolate(
      std::array<size_t, NumberOfVariablesToInterpolate>&
          variables_to_interpolate,
      std::array<double, Dimension>& target_points) const {
    return interpolate(variables_to_interpolate, target_points,
                       std::make_index_sequence<Dimension>{});
  }

  MultiLinearSpanInterpolation() = default;

  MultiLinearSpanInterpolation(
      std::array<gsl::span<const double>, Dimension> x_,
      gsl::span<const double> y_, Index<Dimension> number_of_points__);

  double lower_bound(const size_t which_dimension) const {
    return x_[which_dimension][0];
  }

  double upper_bound(const size_t which_dimension) const {
    return x_[which_dimension][number_of_points_[which_dimension] - 1];
  }

 private:
  /// Allow extrapolation above table bounds.
  std::array<bool, Dimension> allow_extrapolation_below_data_;
  /// Allow extrapolation below table bounds.
  std::array<bool, Dimension> allow_extrapolation_abov_data_;
  /// Inverse pacing of the table. Only used for uniform grids
  std::array<double, Dimension> inverse_spacing_;
  /// Spacing of the table. Only used for uniform grids
  std::array<double, Dimension> spacing_;
  /// Number of points per dimension
  Index<Dimension> number_of_points_;

  using DataPointer = gsl::span<const double>;
  /// X values of the table. Only used if allocated
  std::array<DataPointer, Dimension> x_;
  /// Y values of the table. Only used if allocated
  DataPointer y_;

  /// Determine relative index bracketing for non-uniform _spacing
  size_t find_index_general(const size_t which_dimension,
                            const double& target_points) const;

  /// Determine relative index bracketing for non-uniform _spacing
  size_t find_index_uniform(const size_t which_dimension,
                            const double& target_points) const;

  size_t find_index(const size_t which_dimension,
                    const double& target_points) const {
    if constexpr (UniformSpacing) {
      return find_index_uniform(which_dimension, target_points);
    } else {
      return find_index_general(which_dimension, target_points);
    }
  }
};

template <size_t Dimension, size_t NumberOfVariables, bool UniformSpacing>
size_t MultiLinearSpanInterpolation<
    Dimension, NumberOfVariables,
    UniformSpacing>::find_index_general(const size_t which_dimension,
                                        const double& target_points) const {
  size_t lower_index_bound = 0;
  size_t upper_index_bound = number_of_points_[which_dimension] - 1;

  ASSERT((target_points > x_[which_dimension][lower_index_bound]) or
             allow_extrapolation_below_data_[which_dimension],
         "Interpolation exceeds lower table bounds");
  ASSERT((target_points < x_[which_dimension][upper_index_bound]) or
             allow_extrapolation_abov_data_[which_dimension],
         "Interpolation exceeds upper table bounds");

  while (upper_index_bound > 1 + lower_index_bound) {
    size_t current_index =
        lower_index_bound + (upper_index_bound - lower_index_bound) / 2;
    if (target_points < x_[which_dimension][current_index]) {
      upper_index_bound = current_index;
    } else {
      lower_index_bound = current_index;
    }
  }
  return lower_index_bound;
}

/// Determine relative index bracketing for non-uniform _spacing
template <size_t Dimension, size_t NumberOfVariables, bool UniformSpacing>
size_t MultiLinearSpanInterpolation<
    Dimension, NumberOfVariables,
    UniformSpacing>::find_index_uniform(const size_t which_dimension,
                                        const double& target_points) const {
  // Compute coordinate relative to lowest table bound
  const auto relative_coordinate = (target_points - x_[which_dimension][0]);

  auto current_index = static_cast<size_t>(relative_coordinate *
                                           inverse_spacing_[which_dimension]);

  // We are exceeding the table bounds:
  // Use linear extrapolation based of the lowest
  // two points in the table
  ASSERT(allow_extrapolation_below_data_[which_dimension] or
             UNLIKELY(relative_coordinate >= 0.),
         "Interpolation exceeds lower table bounds.");

  // We are exceeding the table bounds:
  // Use linear extrapolation based of the highest
  // two points in the table

  ASSERT(allow_extrapolation_abov_data_[which_dimension] or
             UNLIKELY(current_index + 1 < number_of_points_[which_dimension]),
         "Interpolation exceeds upper table bounds.");

  // Enforce index ranges
  current_index = std::min(number_of_points_[which_dimension] - 2,
                           std::max(0ul, current_index));

  return current_index;
}

/// Compute interpolation weights for 1D tables
template <size_t Dimension, size_t NumberOfVariables, bool UniformSpacing>
auto MultiLinearSpanInterpolation<Dimension, NumberOfVariables,
                                  UniformSpacing>::get_weights(const double x1)
    const -> Weight<1> {
  // Relative normalized coordinate in x-direction
  double xx;

  Weight<1> weights;

  auto index = Index<1>(find_index(0, x1));

  if constexpr (UniformSpacing) {
    xx = (x1 - x_[0][index[0]]) * inverse_spacing_[0];
  } else {
    xx = (x1 - x_[0][index[0]]) / (x_[0][index[0] + 1] - x_[0][index[0]]);
  }

  // Note: first index varies fastest
  weights.weights = std::array<double, 2>({
      (1. - xx),  // 0
      xx,         // 1
  });

  // Compute indices

  weights.index[0] = index[0];
  weights.index[1] = index[0] + 1;

  return weights;
}

/// Compute interpolation weights for 2D tables

template <size_t Dimension, size_t NumberOfVariables, bool UniformSpacing>
auto MultiLinearSpanInterpolation<Dimension, NumberOfVariables,
                                  UniformSpacing>::get_weights(const double x1,
                                                               const double x2)
    const -> Weight<2> {
  Weight<2> weights;

  auto index = Index<2>(find_index(0, x1), find_index(1, x2));

  // Relative normalized coordinate in x,y-direction
  double xx, yy;
  if constexpr (UniformSpacing) {
    xx = (x1 - x_[0][index[0]]) * inverse_spacing_[0];
    yy = (x2 - x_[1][index[1]]) * inverse_spacing_[1];
  } else {
    xx = (x1 - x_[0][index[0]]) / (x_[0][index[0] + 1] - x_[0][index[0]]);
    yy = (x2 - x_[1][index[1]]) / (x_[1][index[1] + 1] - x_[1][index[1]]);
  }

  // Note: first index varies fastest
  weights.weights = std::array<double, 4>({
      (1. - xx) * (1. - yy),  // 00
      xx * (1. - yy),         // 10
      (1. - xx) * yy,         // 01
      xx * yy,                // 11
  });

  // Compute indices
  //
  for (size_t j = 0; j < 2; ++j) {
    for (size_t i = 0; i < 2; ++i) {
      auto tmp_index = Index<2>(index[0] + i, index[1] + j);
      weights.index[i + 2 * j] = collapsed_index(tmp_index, number_of_points_);
    }
  }

  return weights;
}

/// Compute interpolation weights for 3D tables
template <size_t Dimension, size_t NumberOfVariables, bool UniformSpacing>
auto MultiLinearSpanInterpolation<Dimension, NumberOfVariables,
                                  UniformSpacing>::get_weights(const double x1,
                                                               const double x2,
                                                               const double x3)
    const -> Weight<3> {
  // Relative normalized coordinate in x,y,z-direction
  double xx, yy, zz;

  Weight<3> weights;

  auto index =
      Index<3>(find_index(0, x1), find_index(1, x2), find_index(2, x3));

  if constexpr (UniformSpacing) {
    xx = (x1 - x_[0][index[0]]) * inverse_spacing_[0];
    yy = (x2 - x_[1][index[1]]) * inverse_spacing_[1];
    zz = (x3 - x_[2][index[2]]) * inverse_spacing_[2];
  } else {
    xx = (x1 - x_[0][index[0]]) / (x_[0][index[0] + 1] - x_[0][index[0]]);
    yy = (x2 - x_[1][index[1]]) / (x_[1][index[1] + 1] - x_[1][index[1]]);
    zz = (x3 - x_[2][index[2]]) / (x_[2][index[2] + 1] - x_[2][index[2]]);
  }

  // Note: first index varies fastest
  weights.weights = std::array<double, 8>({
      (1. - xx) * (1. - yy) * (1. - zz),  // 000
      xx * (1. - yy) * (1. - zz),         // 100
      (1. - xx) * yy * (1. - zz),         // 010
      xx * yy * (1. - zz),                // 110
      (1. - xx) * (1. - yy) * zz,         // 001
      xx * (1. - yy) * zz,                // 101
      (1. - xx) * yy * zz,                // 011
      xx * yy * zz,                       // 111
  });

  // Compute indices
  //
  for (size_t k = 0; k < 2; ++k) {
    for (size_t j = 0; j < 2; ++j) {
      for (size_t i = 0; i < 2; ++i) {
        auto tmp_index = Index<3>(index[0] + i, index[1] + j, index[2] + k);
        weights.index[i + 2 * (j + 2 * k)] =
            collapsed_index(tmp_index, number_of_points_);
      }
    }
  }

  return weights;
}

template <size_t Dimension, size_t NumberOfVariables, bool UniformSpacing>
MultiLinearSpanInterpolation<Dimension, NumberOfVariables, UniformSpacing>::
    MultiLinearSpanInterpolation(
        std::array<gsl::span<double const>, Dimension> x,
        gsl::span<double const> y, Index<Dimension> number_of_points)
    : number_of_points_(number_of_points), x_(x), y_(y) {
  for (size_t i = 0; i < Dimension; ++i) {
    spacing_[i] = x_[i][1] - x_[i][0];
    inverse_spacing_[i] = 1. / spacing_[i];
    allow_extrapolation_below_data_[i] = false;
    allow_extrapolation_abov_data_[i] = false;
  }
}

/// Multilinear span interpolation with uniform grid spacing
template <size_t Dimension, size_t NumberOfVariables>
using UniformMultiLinearSpanInterpolation =
    MultiLinearSpanInterpolation<Dimension, NumberOfVariables, true>;

/// Multilinear span interpolation with non-uniform grid spacing
template <size_t Dimension, size_t NumberOfVariables>
using GeneralMultiLinearSpanInterpolation =
    MultiLinearSpanInterpolation<Dimension, NumberOfVariables, false>;

}  // namespace intrp
