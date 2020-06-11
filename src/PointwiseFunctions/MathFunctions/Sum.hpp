// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines MathFunctions::Sum.

#pragma once

#include <array>
#include <memory>
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
class Sum;

/*!
 *  \ingroup MathFunctionsGroup
 *  \brief 1D sum \f$A + B\f$ of two `MathFunction`s \f$A\f$ and \f$B\f$
 *
 *  \details Input file options are: `MathFunctionA` and `MathFunctionB`,
 *  which are two 1-dimensional `MathFunction`s.
 */
template <typename Fr>
class Sum<1, Fr> : public MathFunction<1, Fr> {
 public:
  struct MathFunctionA {
    using type = std::unique_ptr<MathFunction<1, Fr>>;
    static constexpr OptionString help = {"A MathFunction."};
  };

  struct MathFunctionB {
    using type = std::unique_ptr<MathFunction<1, Fr>>;
    static constexpr OptionString help = {"A MathFunction."};
  };

  using options = tmpl::list<MathFunctionA, MathFunctionB>;

  static constexpr OptionString help = {"Sums the input MathFunctions"};

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(MathFunction<1, Fr>),
                                     Sum);  // NOLINT

  explicit Sum(CkMigrateMessage* /*unused*/) noexcept {}

  Sum(std::unique_ptr<MathFunction<1, Fr>> math_function_a,
      std::unique_ptr<MathFunction<1, Fr>> math_function_b) noexcept;

  Sum() = default;
  ~Sum() override = default;
  Sum(const Sum& /*rhs*/) = delete;
  Sum& operator=(const Sum& /*rhs*/) = delete;
  Sum(Sum&& /*rhs*/) noexcept = default;
  Sum& operator=(Sum&& /*rhs*/) noexcept = default;

  double operator()(const double& x) const noexcept override;
  DataVector operator()(const DataVector& x) const noexcept override;
  using MathFunction<1, Fr>::operator();

  double first_deriv(const double& x) const noexcept override;
  DataVector first_deriv(const DataVector& x) const noexcept override;
  using MathFunction<1, Fr>::first_deriv;

  double second_deriv(const double& x) const noexcept override;
  DataVector second_deriv(const DataVector& x) const noexcept override;
  using MathFunction<1, Fr>::second_deriv;

  double third_deriv(const double& x) const noexcept override;
  DataVector third_deriv(const DataVector& x) const noexcept override;
  using MathFunction<1, Fr>::third_deriv;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

 private:
  std::unique_ptr<MathFunction<1, Fr>> math_function_a_;
  std::unique_ptr<MathFunction<1, Fr>> math_function_b_;

  template <typename T>
  T apply_call_operator(const T& x) const noexcept;
  template <typename T>
  T apply_first_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_second_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_third_deriv(const T& x) const noexcept;
};

/*!
 *  \ingroup MathFunctionsGroup
 *  \brief N-dimensional sum \f$A + B\f$ of two `MathFunction`s \f$A\f$ and
 * \f$B\f$
 *
 *  \details Input file options are: `MathFunctionA` and `MathFunctionB`,
 *  which are two N-dimensional `MathFunction`s.
 */
template <size_t VolumeDim, typename Fr>
class Sum : public MathFunction<VolumeDim, Fr> {
 public:
  struct MathFunctionA {
    using type = std::unique_ptr<MathFunction<VolumeDim, Fr>>;
    static constexpr OptionString help = {"A MathFunction."};
  };

  struct MathFunctionB {
    using type = std::unique_ptr<MathFunction<VolumeDim, Fr>>;
    static constexpr OptionString help = {"A MathFunction."};
  };
  using options = tmpl::list<MathFunctionA, MathFunctionB>;

  static constexpr OptionString help = {
      "Applies a Sum function to the input coordinates"};

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(MathFunction<VolumeDim, Fr>),
                                     Sum);  // NOLINT

  explicit Sum(CkMigrateMessage* /*unused*/) noexcept {}

  Sum(std::unique_ptr<MathFunction<VolumeDim, Fr>> math_function_a,
      std::unique_ptr<MathFunction<VolumeDim, Fr>> math_function_b) noexcept;

  Sum() = default;
  ~Sum() override = default;
  Sum(const Sum& /*rhs*/) = delete;
  Sum& operator=(const Sum& /*rhs*/) = delete;
  Sum(Sum&& /*rhs*/) noexcept = default;
  Sum& operator=(Sum&& /*rhs*/) noexcept = default;

  Scalar<double> operator()(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept override;
  Scalar<DataVector> operator()(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept override;

  tnsr::i<double, VolumeDim, Fr> first_deriv(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept override;
  tnsr::i<DataVector, VolumeDim, Fr> first_deriv(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept override;

  tnsr::ii<double, VolumeDim, Fr> second_deriv(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept override;
  tnsr::ii<DataVector, VolumeDim, Fr> second_deriv(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept override;

  tnsr::iii<double, VolumeDim, Fr> third_deriv(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept override;
  tnsr::iii<DataVector, VolumeDim, Fr> third_deriv(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept override;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT
 private:
  std::unique_ptr<MathFunction<VolumeDim, Fr>> math_function_a_;
  std::unique_ptr<MathFunction<VolumeDim, Fr>> math_function_b_;

  template <typename T>
  Scalar<T> apply_call_operator(
      const tnsr::I<T, VolumeDim, Fr>& x) const noexcept;
  template <typename T>
  tnsr::i<T, VolumeDim, Fr> apply_first_deriv(
      const tnsr::I<T, VolumeDim, Fr>& x) const noexcept;
  template <typename T>
  tnsr::ii<T, VolumeDim, Fr> apply_second_deriv(
      const tnsr::I<T, VolumeDim, Fr>& x) const noexcept;
  template <typename T>
  tnsr::iii<T, VolumeDim, Fr> apply_third_deriv(
      const tnsr::I<T, VolumeDim, Fr>& x) const noexcept;
};
}  // namespace MathFunctions

/// \cond
template <size_t VolumeDim, typename Fr>
PUP::able::PUP_ID MathFunctions::Sum<VolumeDim, Fr>::my_PUP_ID = 0;  // NOLINT

template <typename Fr>
PUP::able::PUP_ID MathFunctions::Sum<1, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
