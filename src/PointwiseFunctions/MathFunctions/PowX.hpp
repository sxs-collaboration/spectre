// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines MathFunctions::PowX.

#pragma once

#include "Options/Options.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"

namespace MathFunctions {

/*!
 * \ingroup MathFunctions
 * \brief Power of X \f$f(x)=x^X\f$
 */
class PowX : public MathFunction<1> {
 public:
  static constexpr OptionString_t help = {
      "Raises the input value to a given power"};

  PowX(int power, const OptionContext& context) noexcept;

  explicit PowX(int power) noexcept;

  double operator()(const double& x) const noexcept override;
  DataVector operator()(const DataVector& x) const noexcept override;

  double first_deriv(const double& x) const noexcept override;
  DataVector first_deriv(const DataVector& x) const noexcept override;

  double second_deriv(const double& x) const noexcept override;
  DataVector second_deriv(const DataVector& x) const noexcept override;

  struct Power {
    using type = int;
    static constexpr OptionString_t help = {
        "The power that the double is raised to."};
  };
  using options = tmpl::list<Power>;

 private:
  template <typename T>
  T apply_call_operator(const T& x) const noexcept;
  template <typename T>
  T apply_first_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_second_deriv(const T& x) const noexcept;

  const double power_;
};
}  // namespace MathFunctions
