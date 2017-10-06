// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines MathFunctions::Gaussian.

#pragma once

#include "Options/Options.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"

namespace MathFunctions {

/*!
 *  \ingroup MathFunctions
 *  \brief Gaussian \f$f = A \exp\left(-\frac{(x-x_0)^2}{w^2}\right)\f$.
 *
 *  \details Input file options are: Amplitude, Width, and Center
 */
class Gaussian : public MathFunction<1> {
 public:
  struct Amplitude {
    using type = double;
    static constexpr OptionString_t help = {"The amplitude."};
  };

  struct Width {
    using type = double;
    static constexpr OptionString_t help = {"The width."};
    static type lower_bound() { return 0.; }
  };

  struct Center {
    using type = double;
    static constexpr OptionString_t help = {"The center."};
  };
  using options = tmpl::list<Amplitude, Width, Center>;

  static constexpr OptionString_t help = {
      "Applies a Gaussian function to the input value"};

  Gaussian(double amplitude, double width, double center,
           const OptionContext& context = {}) noexcept;

  double operator()(const double& x) const noexcept override;
  DataVector operator()(const DataVector& x) const noexcept override;

  double first_deriv(const double& x) const noexcept override;
  DataVector first_deriv(const DataVector& x) const noexcept override;

  double second_deriv(const double& x) const noexcept override;
  DataVector second_deriv(const DataVector& x) const noexcept override;

 private:
  template <typename T>
  T apply_call_operator(const T& x) const noexcept;
  template <typename T>
  T apply_first_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_second_deriv(const T& x) const noexcept;

  const double amplitude_;
  const double inverse_width_;
  const double center_;
};
}  // namespace MathFunctions
