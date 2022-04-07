// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <gsl/gsl_multifit.h>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace intrp {
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief A linear least squares solver class
 *
 * A wrapper class for the gsl linear least squares solver which determines
 * the coefficients of best fit for a polynomial of order `Order` given a set
 * of data points `x_values` and `y_values` representing some function y(x).
 * Note that the `interpolate` function requires a set of fit coefficients,
 * which can be obtained by first calling the `fit_coefficients` function.
 *
 * The details of the linear least squares solver can be seen here:
 * [GSL documentation](https://www.gnu.org/software/gsl/doc/html/lls.html#).
 *
 * \warning This class is not thread-safe, the reason for this is because
 * making the buffers used in the class thread-local would be a large use
 * of memory. This class should be treated as a function that internally
 * stores workspace variables to reduce memory allocations.
 */

template <size_t Order>
class LinearLeastSquares {
 public:
  LinearLeastSquares(const size_t num_observations);

  LinearLeastSquares() = default;
  LinearLeastSquares(const LinearLeastSquares& /*rhs*/) = delete;
  LinearLeastSquares& operator=(const LinearLeastSquares& /*rhs*/) = delete;
  LinearLeastSquares(LinearLeastSquares&& /*rhs*/) = default;
  LinearLeastSquares& operator=(LinearLeastSquares&& rhs) = default;
  ~LinearLeastSquares();

  double interpolate(const std::array<double, Order + 1> coefficients,
                     const double x_to_interp_to);
  template <typename T>
  std::array<double, Order + 1> fit_coefficients(const T& x_values,
                                                 const T& y_values);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  size_t num_observations_;
  gsl_matrix* X = nullptr;
  gsl_matrix* covariance_matrix_ = nullptr;
  gsl_vector* y = nullptr;
  gsl_vector* c = nullptr;
};
}  // namespace intrp
