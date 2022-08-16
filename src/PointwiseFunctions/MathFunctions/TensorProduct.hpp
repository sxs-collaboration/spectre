// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines a tensor product of one-dimensional MathFunctions

#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"

namespace MathFunctions {

/// \ingroup MathFunctionsGroup
/// \brief a tensor product of one-dimensional MathFunctions
template <size_t Dim>
class TensorProduct {
 public:
  TensorProduct(double scale,
                std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>,
                           Dim>&& functions);
  TensorProduct() = default;
  TensorProduct(const TensorProduct& other);
  TensorProduct(TensorProduct&&) = default;
  TensorProduct& operator=(const TensorProduct& other);
  TensorProduct& operator=(TensorProduct&&) = default;
  ~TensorProduct() = default;

  /// The value of the function
  template <typename T>
  Scalar<T> operator()(const tnsr::I<T, Dim>& x) const;

  /// The partial derivatives of the function
  template <typename T>
  tnsr::i<T, Dim> first_derivatives(const tnsr::I<T, Dim>& x) const;

  /// The second partial derivatives of the function
  template <typename T>
  tnsr::ii<T, Dim> second_derivatives(const tnsr::I<T, Dim>& x) const;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const TensorProduct<LocalDim>& lhs,
                         const TensorProduct<LocalDim>& rhs);
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator!=(const TensorProduct<LocalDim>& lhs,
                         const TensorProduct<LocalDim>& rhs);
  double scale_{1.0};
  std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, Dim> functions_;
};
}  // namespace MathFunctions
