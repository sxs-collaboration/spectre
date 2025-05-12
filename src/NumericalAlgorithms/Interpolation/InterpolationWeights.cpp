// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/InterpolationWeights.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace intrp {
template <typename TargetDataType>
Matrix fornberg_interpolation_matrix(const TargetDataType& x_target,
                                     const DataVector& x_source) {
  Matrix result{get_size(x_target), x_source.size()};
  const size_t n_source_pts = x_source.size();
  for (size_t k = 0; k < get_size(x_target); ++k) {
    double c1 = 1.0;
    double c4 = x_source[0] - get_element(x_target, k);
    result(k, 0) = 1.0;
    for (size_t i = 1; i < n_source_pts; ++i) {
      double c2 = 1.0;
      const double c5 = c4;
      c4 = x_source[i] - get_element(x_target, k);
      for (size_t j = 0; j < i; ++j) {
        const double c3 = x_source[i] - x_source[j];
        c2 *= c3;
        if (j + 1 == i) {
          result(k, i) = -c1 * c5 * result(k, i - 1) / c2;
        }
        result(k, j) = c4 * result(k, j) / c3;
      }
      c1 = c2;
    }
  }
  return result;
}

template <typename TargetDataType>
Matrix fourier_interpolation_matrix(const TargetDataType& x_target,
                                    const size_t n_source_points) {
  if (n_source_points == 1) {
    return Matrix{get_size(x_target), 1, 1.0};
  }
  DataVector x_source{n_source_points,
                      2.0 * M_PI / static_cast<double>(n_source_points)};
  for (size_t i = 0; i < n_source_points; ++i) {
    x_source[i] *= static_cast<double>(i);
  }
  Matrix result{get_size(x_target), n_source_points,
                1.0 / static_cast<double>(n_source_points)};
  const bool n_source_points_is_even = n_source_points % 2 == 0;
  for (size_t k = 0; k < get_size(x_target); ++k) {
    for (size_t i = 0; i < n_source_points; ++i) {
      double c0 = 1.0;
      double c1 = cos(get_element(x_target, k) - x_source[i]);
      const double cdx2 = 2.0 * c1;
      double sum = c1;
      for (size_t j = 2; j <= n_source_points / 2; ++j) {
        const double tmp = c1;
        c1 = cdx2 * c1 - c0;
        c0 = tmp;
        sum += c1;
      }
      result(k, i) *=
          n_source_points_is_even ? 2.0 * sum + 1.0 - c1 : 2.0 * sum + 1.0;
    }
  }
  return result;
}

// Generate instantiations

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template Matrix fornberg_interpolation_matrix(const DTYPE(data) & x_target, \
                                                const DataVector& x_source);  \
  template Matrix fourier_interpolation_matrix(const DTYPE(data) & x_target,  \
                                               const size_t n_source_points);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace intrp
