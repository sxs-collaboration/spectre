// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines base-class MathFunction.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \ingroup MathFunctionsGroup
/// Holds classes implementing MathFunction (functions \f$R^n \to R\f$).
namespace MathFunctions {
template <size_t VolumeDim, typename Fr>
class Constant;
template <size_t VolumeDim, typename Fr>
class Gaussian;
template <size_t VolumeDim, typename Fr>
class PowX;
template <size_t VolumeDim, typename Fr>
class Sinusoid;
template <size_t VolumeDim, typename Fr>
class Sum;
}  // namespace MathFunctions

/*!
 * \ingroup MathFunctionsGroup
 * Encodes a function \f$R^n \to R\f$ where n is `VolumeDim`.
 */
template <size_t VolumeDim, typename Fr>
class MathFunction;

/*!
 * \ingroup MathFunctionsGroup
 * Encodes a function \f$R^n \to R\f$ where n is `VolumeDim` and where the
 * function input (i.e., the spatial coordinates) is given as a rank-1 tensor.
 */
template <size_t VolumeDim, typename Fr>
class MathFunction : public PUP::able {
 public:
  using creatable_classes = tmpl::list<MathFunctions::Constant<VolumeDim, Fr>,
                                       MathFunctions::Gaussian<VolumeDim, Fr>,
                                       MathFunctions::Sum<VolumeDim, Fr>>;
  constexpr static size_t volume_dim = VolumeDim;
  using frame = Fr;

  WRAPPED_PUPable_abstract(MathFunction);  // NOLINT

  MathFunction() = default;
  MathFunction(const MathFunction& /*rhs*/) = delete;
  MathFunction& operator=(const MathFunction& /*rhs*/) = delete;
  MathFunction(MathFunction&& /*rhs*/) noexcept = default;
  MathFunction& operator=(MathFunction&& /*rhs*/) noexcept = default;
  ~MathFunction() override = default;

  //@{
  /// Returns the value of the function at the coordinate 'x'.
  virtual Scalar<double> operator()(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept = 0;
  virtual Scalar<DataVector> operator()(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept = 0;
  //@}

  //@{
  /// Returns the first partial derivatives of the function at 'x'.
  virtual tnsr::i<double, VolumeDim, Fr> first_deriv(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept = 0;
  virtual tnsr::i<DataVector, VolumeDim, Fr> first_deriv(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept = 0;
  //@}

  //@{
  /// Returns the second partial derivatives of the function at 'x'.
  virtual tnsr::ii<double, VolumeDim, Fr> second_deriv(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept = 0;
  virtual tnsr::ii<DataVector, VolumeDim, Fr> second_deriv(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept = 0;
  //@}

  //@{
  /// Returns the third partial derivatives of the function at 'x'.
  virtual tnsr::iii<double, VolumeDim, Fr> third_deriv(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept = 0;
  virtual tnsr::iii<DataVector, VolumeDim, Fr> third_deriv(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept = 0;
  //@}
};

/*!
 * \ingroup MathFunctionsGroup
 * Partial template specialization of MathFunction which encodes a
 * function \f$R \to R\f$. In this 1D specialization, the input and output can
 * be `Tensors`, `doubles`, or `DataVectors`.
 */
template <typename Fr>
class MathFunction<1, Fr> : public PUP::able {
 public:
  using creatable_classes =
      tmpl::list<MathFunctions::Constant<1, Fr>, MathFunctions::Gaussian<1, Fr>,
                 MathFunctions::PowX<1, Fr>, MathFunctions::Sinusoid<1, Fr>,
                 MathFunctions::Sum<1, Fr>>;
  constexpr static size_t volume_dim = 1;
  using frame = Fr;

  WRAPPED_PUPable_abstract(MathFunction);  // NOLINT

  MathFunction() = default;
  MathFunction(const MathFunction& /*rhs*/) = delete;
  MathFunction& operator=(const MathFunction& /*rhs*/) = delete;
  MathFunction(MathFunction&& /*rhs*/) noexcept = default;
  MathFunction& operator=(MathFunction&& /*rhs*/) noexcept = default;
  ~MathFunction() override = default;

  /// Returns the function value at the coordinate 'x'
  virtual double operator()(const double& x) const noexcept = 0;
  virtual DataVector operator()(const DataVector& x) const noexcept = 0;
  Scalar<double> operator()(const tnsr::I<double, 1, Fr>& x) const noexcept {
    return Scalar<double>{{{operator()(get<0>(x))}}};
  }
  Scalar<DataVector> operator()(
      const tnsr::I<DataVector, 1, Fr>& x) const noexcept {
    return Scalar<DataVector>{{{operator()(get<0>(x))}}};
  }

  /// Returns the first derivative at 'x'
  virtual double first_deriv(const double& x) const noexcept = 0;
  virtual DataVector first_deriv(const DataVector& x) const noexcept = 0;
  tnsr::i<double, 1, Fr> first_deriv(
      const tnsr::I<double, 1, Fr>& x) const noexcept {
    auto result = make_with_value<tnsr::i<double, 1, Fr>>(get<0>(x), 0.0);
    get<0>(result) = first_deriv(get<0>(x));
    return result;
  }
  tnsr::i<DataVector, 1, Fr> first_deriv(
      const tnsr::I<DataVector, 1, Fr>& x) const noexcept {
    auto result = make_with_value<tnsr::i<DataVector, 1, Fr>>(get<0>(x), 0.0);
    get<0>(result) = first_deriv(get<0>(x));
    return result;
  }

  /// Returns the second derivative at 'x'
  virtual double second_deriv(const double& x) const noexcept = 0;
  virtual DataVector second_deriv(const DataVector& x) const noexcept = 0;
  tnsr::ii<double, 1, Fr> second_deriv(
      const tnsr::I<double, 1, Fr>& x) const noexcept {
    auto result = make_with_value<tnsr::ii<double, 1, Fr>>(get<0>(x), 0.0);
    get<0, 0>(result) = second_deriv(get<0>(x));
    return result;
  }
  tnsr::ii<DataVector, 1, Fr> second_deriv(
      const tnsr::I<DataVector, 1, Fr>& x) const noexcept {
    auto result = make_with_value<tnsr::ii<DataVector, 1, Fr>>(get<0>(x), 0.0);
    get<0, 0>(result) = second_deriv(get<0>(x));
    return result;
  }

  /// Returns the third derivative at 'x'
  virtual double third_deriv(const double& x) const noexcept = 0;
  virtual DataVector third_deriv(const DataVector& x) const noexcept = 0;
  tnsr::iii<double, 1, Fr> third_deriv(
      const tnsr::I<double, 1, Fr>& x) const noexcept {
    auto result = make_with_value<tnsr::iii<double, 1, Fr>>(get<0>(x), 0.0);
    get<0, 0, 0>(result) = third_deriv(get<0>(x));
    return result;
  }
  tnsr::iii<DataVector, 1, Fr> third_deriv(
      const tnsr::I<DataVector, 1, Fr>& x) const noexcept {
    auto result = make_with_value<tnsr::iii<DataVector, 1, Fr>>(get<0>(x), 0.0);
    get<0, 0, 0>(result) = third_deriv(get<0>(x));
    return result;
  }
};

#include "PointwiseFunctions/MathFunctions/Constant.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
#include "PointwiseFunctions/MathFunctions/Sum.hpp"
