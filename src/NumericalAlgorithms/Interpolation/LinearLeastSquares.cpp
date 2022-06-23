// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/LinearLeastSquares.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <gsl/gsl_spline.h>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {
template <size_t Order>
LinearLeastSquares<Order>::LinearLeastSquares(const size_t num_observations)
    : num_observations_(num_observations) {
  X = gsl_matrix_alloc(num_observations_, Order + 1);
  y = gsl_vector_alloc(num_observations_);
  c = gsl_vector_alloc(Order + 1);
  covariance_matrix_ = gsl_matrix_alloc(Order + 1, Order + 1);
}

template <size_t Order>
LinearLeastSquares<Order>::~LinearLeastSquares() {
  gsl_matrix_free(X);
  gsl_vector_free(y);
  gsl_vector_free(c);
  gsl_matrix_free(covariance_matrix_);
}

template <size_t Order>
double LinearLeastSquares<Order>::interpolate(
    const std::array<double, Order + 1> coefficients,
    const double x_to_interp_to) {
  double result = 0;
  for (size_t i = 0; i < Order + 1; i++) {
    result += gsl::at(coefficients, i) * pow(x_to_interp_to, i);
  }
  return result;
}

template <size_t Order>
template <typename T>
std::array<double, Order + 1> LinearLeastSquares<Order>::fit_coefficients(
    const T& x_values, const T& y_values) {
  std::array<double, Order + 1> coefficients{{}};
  for (size_t i = 0; i < num_observations_; i++) {
    for (size_t j = 0; j < Order + 1; j++) {
      gsl_matrix_set(X, i, j, pow(x_values[i], j));
    }
    gsl_vector_set(y, i, y_values[i]);
  }

  gsl_multifit_linear_workspace* work =
      gsl_multifit_linear_alloc(num_observations_, Order + 1);
  double chisq = 0.0;
  gsl_multifit_linear(X, y, c, covariance_matrix_, &chisq, work);
  gsl_multifit_linear_free(work);

  for (size_t i = 0; i < Order + 1; i++) {
    gsl::at(coefficients, i) = gsl_vector_get(c, i);
  }
  return coefficients;
}

template <size_t Order>
template <typename T>
std::vector<std::array<double, Order + 1>>
LinearLeastSquares<Order>::fit_coefficients(const T& x_values,
                                            const std::vector<T>& y_values) {
  ASSERT(not y_values.empty(), "Must have non-zero series");
  std::vector<std::array<double, Order + 1>> fit_coeffs{};
  for (size_t curve_index = 0; curve_index < y_values.size(); ++curve_index) {
    ASSERT(x_values.size() == y_values[curve_index].size(),
           "The x_values and y_values must be of the same size");
    fit_coeffs.push_back(fit_coefficients(x_values, y_values[curve_index]));
  }
  return fit_coeffs;
}

template <size_t Order>
void LinearLeastSquares<Order>::pup(PUP::er& p) {
  p | num_observations_;
  if (p.isUnpacking()) {
    X = gsl_matrix_alloc(num_observations_, Order + 1);
    y = gsl_vector_alloc(num_observations_);
    c = gsl_vector_alloc(Order + 1);
    covariance_matrix_ = gsl_matrix_alloc(Order + 1, Order + 1);
  }
}

// Explicit instantiations
#define ORDER(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE_ORDER(_, data) \
  template class LinearLeastSquares<ORDER(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE_ORDER, (1, 2, 3, 4))

#undef INSTANTIATE_ORDER
#define INSTANTIATE_DTYPE1(_, data)                  \
  template std::array<double, ORDER(data) + 1>       \
  LinearLeastSquares<ORDER(data)>::fit_coefficients( \
      const DTYPE(data) & x_values, const DTYPE(data) & y_values);

GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE1, (1, 2, 3, 4),
                        (std::vector<double>, DataVector, ModalVector,
                         gsl::span<double>, std::deque<double>))

#define INSTANTIATE_DTYPE2(_, data)                         \
  template std::vector<std::array<double, ORDER(data) + 1>> \
  LinearLeastSquares<ORDER(data)>::fit_coefficients(        \
      const DTYPE(data) & x_values, const std::vector<DTYPE(data)>& y_values);

GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE2, (1, 2, 3, 4),
                        (std::vector<double>, DataVector, ModalVector,
                         std::deque<double>))
#undef ORDER
#undef DTYPE

}  // namespace intrp
