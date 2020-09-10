// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines MathFunctions::PowX.

#pragma once

#include <pup.h>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace MathFunctions {
template <size_t VolumeDim, typename Fr>
class PowX;

/*!
 * \ingroup MathFunctionsGroup
 * \brief Power of X \f$f(x)=x^X\f$
 */
template <typename Fr>
class PowX<1, Fr> : public MathFunction<1, Fr> {
 public:
  struct Power {
    using type = int;
    static constexpr Options::String help = {
        "The power that the double is raised to."};
  };
  using options = tmpl::list<Power>;

  static constexpr Options::String help = {
      "Raises the input value to a given power"};

  PowX() = default;
  ~PowX() override = default;
  PowX(const PowX& /*rhs*/) = delete;
  PowX& operator=(const PowX& /*rhs*/) = delete;
  PowX(PowX&& /*rhs*/) noexcept = default;
  PowX& operator=(PowX&& /*rhs*/) noexcept = default;

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(MathFunction<1, Fr>),
                                     PowX);  // NOLINT

  explicit PowX(int power) noexcept;

  explicit PowX(CkMigrateMessage* /*unused*/) noexcept {}

  double operator()(const double& x) const noexcept override;
  DataVector operator()(const DataVector& x) const noexcept override;

  double first_deriv(const double& x) const noexcept override;
  DataVector first_deriv(const DataVector& x) const noexcept override;

  double second_deriv(const double& x) const noexcept override;
  DataVector second_deriv(const DataVector& x) const noexcept override;

  double third_deriv(const double& x) const noexcept override;
  DataVector third_deriv(const DataVector& x) const noexcept override;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

 private:
  double power_{};
  friend bool operator==(const PowX& lhs, const PowX& rhs) noexcept {
    return lhs.power_ == rhs.power_;
  }

  template <typename T>
  T apply_call_operator(const T& x) const noexcept;
  template <typename T>
  T apply_first_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_second_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_third_deriv(const T& x) const noexcept;
};

template <typename Fr>
bool operator!=(const PowX<1, Fr>& lhs, const PowX<1, Fr>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace MathFunctions

/// \cond
template <typename Fr>
PUP::able::PUP_ID MathFunctions::PowX<1, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
