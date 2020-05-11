// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <random>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

/// \ingroup TestingFrameworkGroup
/// \brief Construct a spatial vector in a given magnitude range
///
/// The magnitude is computed with respect to the given metric, where the metric
/// is assumed to have positive signature.
template <typename DataType, size_t Dim, UpLo Ul, typename Fr = Frame::Inertial,
          Requires<(Ul == UpLo::Up)> = nullptr>
tnsr::I<DataType, Dim, Fr> make_random_vector_in_magnitude_range(
    const gsl::not_null<std::mt19937*> nn_generator,
    const tnsr::ii<DataType, Dim, Fr>& metric, const double min_magnitude,
    const double max_magnitude) noexcept {
  if (min_magnitude < 0) {
    ERROR("min_magnitude < 0. Magnitude must be non-negative");
  }
  if (max_magnitude < 0) {
    ERROR("max_magnitude < 0. Magnitude must be non-negative");
  }

  // generate distribution
  std::uniform_real_distribution<> dist_magnitude(min_magnitude, max_magnitude);
  const auto magnitude = make_with_random_values<Scalar<DataType>>(
      nn_generator, make_not_null(&dist_magnitude), metric);

  // construct vector
  tnsr::I<DataType, Dim, Fr> x_up = random_unit_normal(nn_generator, metric);

  for (size_t i = 0; i < Dim; i++) {
    x_up.get(i) *= magnitude.get();
  }

  return x_up;
}

template <typename DataType, size_t Dim, UpLo Ul, typename Fr = Frame::Inertial,
          Requires<(Ul == UpLo::Lo)> = nullptr>
tnsr::i<DataType, Dim, Fr> make_random_vector_in_magnitude_range(
    const gsl::not_null<std::mt19937*> nn_generator,
    const tnsr::ii<DataType, Dim, Fr>& metric, const double min_magnitude,
    const double max_magnitude) noexcept {
  const tnsr::I<DataType, Dim, Fr> x_up =
      make_random_vector_in_magnitude_range<DataType, Dim, UpLo::Up, Fr>(
          nn_generator, metric, min_magnitude, max_magnitude);

  return raise_or_lower_index(x_up, metric);
}

/// \ingroup TestingFrameworkGroup
/// \brief Construct a spatial vector in a given magnitude range
///
/// The magnitude is computed with respect to the flat space Euclidian metric.
template <typename DataType, size_t Dim, UpLo Ul, typename Fr = Frame::Inertial,
          typename T>
Tensor<DataType, Symmetry<1>, index_list<SpatialIndex<Dim, Ul, Fr>>>
make_random_vector_in_magnitude_range_flat(
    const gsl::not_null<std::mt19937*> nn_generator, const T& used_for_size,
    const double min_magnitude, const double max_magnitude) noexcept {
  // construct flat spatial metric
  tnsr::ii<DataType, Dim, Fr> metric =
      make_with_value<tnsr::ii<DataType, Dim, Fr>>(used_for_size, 0.0);
  for (size_t i = 0; i < Dim; i++) {
    metric.get(i, i) = 1.0;
  }

  return make_random_vector_in_magnitude_range<DataType, Dim, Ul, Fr>(
      nn_generator, metric, min_magnitude, max_magnitude);
}
