// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines a tensor product of one-dimensional MathFunctions

#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
template <size_t Dim>
class MathFunction;
/// \endcond

namespace MathFunctions {

/// \ingroup MathFunctionsGroup
/// \brief a tensor product of one-dimensional MathFunctions
template <size_t Dim>
class TensorProduct {
 public:
  TensorProduct(double scale,
                std::array<std::unique_ptr<MathFunction<1>>, Dim>&& functions);

  TensorProduct(const TensorProduct&) = delete;
  TensorProduct(TensorProduct&&) = default;
  TensorProduct& operator=(const TensorProduct&) = delete;
  TensorProduct& operator=(TensorProduct&&) = default;
  ~TensorProduct() = default;

  /// The value of the function
  template <typename T>
  Scalar<T> operator()(const tnsr::I<T, Dim>& x) const noexcept;

  /// The partial derivatives of the function
  template <typename T>
  tnsr::i<T, Dim> first_derivatives(const tnsr::I<T, Dim>& x) const noexcept;

  /// The second partial derivatives of the function
  template <typename T>
  tnsr::ii<T, Dim> second_derivatives(const tnsr::I<T, Dim>& x) const noexcept;

 private:
  double scale_{1.0};
  std::array<std::unique_ptr<MathFunction<1>>, Dim> functions_;
};
}  // namespace MathFunctions
