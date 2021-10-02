// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <gsl/gsl_spline.h>
#include <memory>
#include <vector>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace intrp {
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief A natural cubic spline interpolation class
 *
 * The class builds a cubic spline interpolant with natural boundary conditions
 * using the `x_values` and `y_values` passed into the constructor. For details
 * on the algorithm see the GSL documentation on `gsl_interp_cspline`.
 *
 * Here is an example how to use this class:
 *
 * \snippet Test_CubicSpline.cpp interpolate_example
 */
class CubicSpline {
 public:
  CubicSpline(std::vector<double> x_values, std::vector<double> y_values);

  CubicSpline() = default;
  CubicSpline(const CubicSpline& /*rhs*/) = delete;
  CubicSpline& operator=(const CubicSpline& /*rhs*/) = delete;
  CubicSpline(CubicSpline&& /*rhs*/) = default;
  CubicSpline& operator=(CubicSpline&& rhs) = default;
  ~CubicSpline() = default;

  double operator()(double x_to_interp_to) const;

  // clang-tidy: no runtime references
  void pup(PUP::er& p);  // NOLINT

 private:
  struct gsl_interp_accel_deleter {
    void operator()(gsl_interp_accel* acc) const;
  };
  struct gsl_spline_deleter {
    void operator()(gsl_spline* spline) const;
  };

  void initialize_interpolant();

  std::vector<double> x_values_;
  std::vector<double> y_values_;
  std::unique_ptr<gsl_interp_accel, gsl_interp_accel_deleter> acc_;
  std::unique_ptr<gsl_spline, gsl_spline_deleter> spline_;
};
}  // namespace intrp
