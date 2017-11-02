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
  struct Power {
    using type = int;
    static constexpr OptionString_t help = {
        "The power that the double is raised to."};
  };
  using options = tmpl::list<Power>;

  static constexpr OptionString_t help = {
      "Raises the input value to a given power"};

  PowX() = default;
  ~PowX() override = default;
  PowX(const PowX& /*rhs*/) = delete;
  PowX& operator=(const PowX& /*rhs*/) = delete;
  PowX(PowX&& /*rhs*/) noexcept = default;
  PowX& operator=(PowX&& /*rhs*/) noexcept = default;

  WRAPPED_PUPable_decl_template(PowX);  // NOLINT

  explicit PowX(int power) noexcept;

  explicit PowX(CkMigrateMessage* /*unused*/) noexcept {}

  double operator()(const double& x) const noexcept override;
  DataVector operator()(const DataVector& x) const noexcept override;

  double first_deriv(const double& x) const noexcept override;
  DataVector first_deriv(const DataVector& x) const noexcept override;

  double second_deriv(const double& x) const noexcept override;
  DataVector second_deriv(const DataVector& x) const noexcept override;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

 private:
  template <typename T>
  T apply_call_operator(const T& x) const noexcept;
  template <typename T>
  T apply_first_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_second_deriv(const T& x) const noexcept;

  double power_{};
};
}  // namespace MathFunctions
