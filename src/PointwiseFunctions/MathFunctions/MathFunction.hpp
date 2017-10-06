// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines base-class MathFunction.

#pragma once

#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Factory.hpp"

/// Holds classes implementing MathFunction (functions \f$R^n \to R\f$).
namespace MathFunctions {
class Gaussian;
class PowX;
class Sinusoid;
}  // namespace MathFunctions

/*!
 *  \ingroup MathFunctions
 *  Encodes a function \f$R^n \to R\f$ where n is `VolumeDim`.
 */
template <size_t VolumeDim>
class MathFunction;

/*!
 * \ingroup MathFunctions
 * Partial template specialization of MathFunction which encodes a
 * function \f$R \to R\f$.
 */
template <>
class MathFunction<1> : public Factory<MathFunction<1>> {
 public:
  using creatable_classes = typelist<
      MathFunctions::Gaussian,
      MathFunctions::PowX
      // ,MathFunctions::Sinusoid
      >;

  MathFunction() = default;
  MathFunction(const MathFunction& /*rhs*/) = delete;
  MathFunction& operator=(const MathFunction& /*rhs*/) = delete;
  MathFunction(MathFunction&& /*rhs*/) noexcept = default;             // NOLINT
  MathFunction& operator=(MathFunction&& /*rhs*/) noexcept = default;  // NOLINT
  virtual ~MathFunction() = default;

  //@{
  /// Returns the function value at the coordinate 'x'.
  virtual double operator()(const double& x) const noexcept = 0;
  virtual DataVector operator()(const DataVector& x) const noexcept = 0;
  //@}

  //@{
  /// Returns the first derivative at 'x'.
  virtual double first_deriv(const double& x) const noexcept = 0;
  virtual DataVector first_deriv(const DataVector& x) const noexcept = 0;
  //@}

  //@{
  /// Returns the second derivative at 'x'.
  virtual double second_deriv(const double& x) const noexcept = 0;
  virtual DataVector second_deriv(const DataVector& x) const noexcept = 0;
  //@}
};

#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
// #include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
