// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearAlgebra/LinearLeastSquares.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_vector.h>
#include <memory>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {
namespace {
template <size_t Order, typename T>
std::array<double, Order + 1> least_squares_impl(
    const gsl::not_null<gsl_matrix*> X,
    const gsl::not_null<gsl_matrix*> covariance_matrix,
    const gsl::not_null<gsl_vector*> y, const gsl::not_null<gsl_vector*> c,
    const gsl::not_null<gsl_multifit_linear_workspace*> work, const T& x_values,
    const T& y_values) {
  ASSERT(x_values.size() == y_values.size(),
         "The x_values and y_values must be of the same size");
  std::array<double, Order + 1> coefficients{{}};
  for (size_t i = 0; i < x_values.size(); i++) {
    for (size_t j = 0; j < Order + 1; j++) {
      gsl_matrix_set(X.get(), i, j, pow(x_values[i], j));
    }
    gsl_vector_set(y.get(), i, y_values[i]);
  }

  double chisq = 0.0;
  gsl_multifit_linear(X.get(), y.get(), c.get(), covariance_matrix.get(),
                      &chisq, work.get());

  for (size_t i = 0; i < Order + 1; i++) {
    gsl::at(coefficients, i) = gsl_vector_get(c.get(), i);
  }
  return coefficients;
}

struct FreeVector {
  void operator()(gsl_vector* const p) { gsl_vector_free(p); }
};

struct FreeMatrix {
  void operator()(gsl_matrix* const p) { gsl_matrix_free(p); }
};

struct FreeWork {
  void operator()(gsl_multifit_linear_workspace* const p) {
    gsl_multifit_linear_free(p);
  }
};
}  // namespace

template <size_t Order, typename T>
std::array<double, Order + 1> linear_least_squares(const T& x_values,
                                                   const T& y_values) {
  const std::unique_ptr<gsl_matrix, FreeMatrix> X(
      gsl_matrix_alloc(x_values.size(), Order + 1));
  const std::unique_ptr<gsl_vector, FreeVector> y(
      gsl_vector_alloc(x_values.size()));
  const std::unique_ptr<gsl_vector, FreeVector> c(gsl_vector_alloc(Order + 1));
  const std::unique_ptr<gsl_matrix, FreeMatrix> covariance_matrix(
      gsl_matrix_alloc(Order + 1, Order + 1));
  const std::unique_ptr<gsl_multifit_linear_workspace, FreeWork> work(
      gsl_multifit_linear_alloc(x_values.size(), Order + 1));
  return least_squares_impl<Order>(X.get(), covariance_matrix.get(), y.get(),
                                   c.get(), work.get(), x_values, y_values);
}

template <size_t Order, typename T>
std::vector<std::array<double, Order + 1>> linear_least_squares(
    const T& x_values, const std::vector<T>& y_values) {
  const std::unique_ptr<gsl_matrix, FreeMatrix> X(
      gsl_matrix_alloc(x_values.size(), Order + 1));
  const std::unique_ptr<gsl_vector, FreeVector> y(
      gsl_vector_alloc(x_values.size()));
  const std::unique_ptr<gsl_vector, FreeVector> c(gsl_vector_alloc(Order + 1));
  const std::unique_ptr<gsl_matrix, FreeMatrix> covariance_matrix(
      gsl_matrix_alloc(Order + 1, Order + 1));
  const std::unique_ptr<gsl_multifit_linear_workspace, FreeWork> work(
      gsl_multifit_linear_alloc(x_values.size(), Order + 1));
  std::vector<std::array<double, Order + 1>> fit_coeffs{};
  for (size_t curve_index = 0; curve_index < y_values.size(); ++curve_index) {
    fit_coeffs.push_back(least_squares_impl<Order>(
        X.get(), covariance_matrix.get(), y.get(), c.get(), work.get(),
        x_values, y_values[curve_index]));
  }
  return fit_coeffs;
}

// Explicit instantiations
#define ORDER(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                       \
  template std::array<double, ORDER(data) + 1>                     \
  linear_least_squares<ORDER(data)>(const DTYPE(data) & x_values,  \
                                    const DTYPE(data) & y_values); \
  template std::vector<std::array<double, ORDER(data) + 1>>        \
  linear_least_squares<ORDER(data)>(const DTYPE(data) & x_values,  \
                                    const std::vector<DTYPE(data)>& y_values);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3, 4),
                        (std::vector<double>, DataVector, ModalVector,
                         gsl::span<double>, std::deque<double>))

#undef INSTANTIATE
#undef DTYPE
#undef ORDER
}  // namespace intrp
