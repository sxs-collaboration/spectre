// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines MathFunctions::Constant.

#pragma once

#include <array>
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
class Constant;

/*!
 *  \ingroup MathFunctionsGroup
 *  \brief Returns a constant value, independent of the input.
 *
 *  \details The input file takes one option: Value, the constant value
 * returned. The function takes input of type `double` or `DataVector` and
 * returns the same type as the input type.
 */
template <typename Fr>
class Constant<1, Fr> : public MathFunction<1, Fr> {
 public:
  struct Value {
    using type = double;
    static constexpr OptionString help = {"The constant value."};
  };

  using options = tmpl::list<Value>;

  static constexpr OptionString help = {
      "Returns a constant independent of the input value"};

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(MathFunction<1, Fr>),
                                     Constant);  // NOLINT

  explicit Constant(CkMigrateMessage* /*unused*/) noexcept {}

  explicit Constant(double value) noexcept;

  Constant() = default;
  ~Constant() override = default;
  Constant(const Constant& /*rhs*/) = delete;
  Constant& operator=(const Constant& /*rhs*/) = delete;
  Constant(Constant&& /*rhs*/) noexcept = default;
  Constant& operator=(Constant&& /*rhs*/) noexcept = default;

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

  double value() const noexcept { return value_; }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

 private:
  double value_{};
  friend bool operator==(const Constant<1, Fr>& lhs,
                         const Constant<1, Fr>& rhs) noexcept {
    return lhs.value_ == rhs.value_;
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

/*!
 *  \ingroup MathFunctionsGroup
 *  \brief Returns a constant value, independent of the input.
 *
 *  \details Input file options are: Value. The function
 *  takes input coordinates of type `tnsr::I<T, VolumeDim, Fr>`, where `T` is
 * e.g. `double` or `DataVector`, `Fr` is a frame (e.g. `Frame::Inertial`), and
 * `VolumeDim` is the dimension of the spatial volume, i.e. 2 or 3. (The case of
 * VolumeDim == 1 is handled specially by Constant<1, T, Frame::Inertial>.)
 */
template <size_t VolumeDim, typename Fr>
class Constant : public MathFunction<VolumeDim, Fr> {
 public:
  struct Value {
    using type = double;
    static constexpr OptionString help = {"The constant value."};
  };

  using options = tmpl::list<Value>;

  static constexpr OptionString help = {
      "Returns a Constant independent of the input coordinates"};

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(MathFunction<VolumeDim, Fr>),
                                     Constant);  // NOLINT

  explicit Constant(CkMigrateMessage* /*unused*/) noexcept {}

  explicit Constant(double value) noexcept;

  Constant() = default;
  ~Constant() override = default;
  Constant(const Constant& /*rhs*/) = delete;
  Constant& operator=(const Constant& /*rhs*/) = delete;
  Constant(Constant&& /*rhs*/) noexcept = default;
  Constant& operator=(Constant&& /*rhs*/) noexcept = default;

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
  double value_{};
  friend bool operator==(const Constant& lhs, const Constant& rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

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

template <size_t VolumeDim, typename Fr>
bool operator!=(const Constant<VolumeDim, Fr>& lhs,
                const Constant<VolumeDim, Fr>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace MathFunctions

/// \cond
template <size_t VolumeDim, typename Fr>
PUP::able::PUP_ID MathFunctions::Constant<VolumeDim, Fr>::my_PUP_ID =
    0;  // NOLINT

template <typename Fr>
PUP::able::PUP_ID MathFunctions::Constant<1, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
