// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines MathFunctions::PowX.

#pragma once

#include <memory>
#include <pup.h>

#include "Options/String.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"  // IWYU pragma: keep
#include "Utilities/Serialization/CharmPupable.hpp"
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

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(MathFunction<1, Fr>),
                                     PowX);  // NOLINT
  std::unique_ptr<MathFunction<1, Fr>> get_clone() const override;

  explicit PowX(int power);

  explicit PowX(CkMigrateMessage* /*unused*/) {}

  double operator()(const double& x) const override;
  DataVector operator()(const DataVector& x) const override;

  double first_deriv(const double& x) const override;
  DataVector first_deriv(const DataVector& x) const override;

  double second_deriv(const double& x) const override;
  DataVector second_deriv(const DataVector& x) const override;

  double third_deriv(const double& x) const override;
  DataVector third_deriv(const DataVector& x) const override;

  bool operator==(const MathFunction<1, Fr>& other) const override;
  bool operator!=(const MathFunction<1, Fr>& other) const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double power_{};

  template <typename T>
  T apply_call_operator(const T& x) const;
  template <typename T>
  T apply_first_deriv(const T& x) const;
  template <typename T>
  T apply_second_deriv(const T& x) const;
  template <typename T>
  T apply_third_deriv(const T& x) const;
};

}  // namespace MathFunctions

/// \cond
template <typename Fr>
PUP::able::PUP_ID MathFunctions::PowX<1, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
