// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace MathFunctions {

template <size_t Dim>
TensorProduct<Dim>::TensorProduct(
    double scale, std::array<std::unique_ptr<MathFunction<1>>, Dim>&& functions)
    : scale_(scale), functions_(std::move(functions)) {}

template <size_t Dim>
template <typename T>
Scalar<T> TensorProduct<Dim>::operator()(const tnsr::I<T, Dim>& x) const
    noexcept {
  auto result = make_with_value<Scalar<T>>(x, scale_);
  for (size_t d = 0; d < Dim; ++d) {
    result.get() *= gsl::at(functions_, d)->operator()(x.get(d));
  }
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::i<T, Dim> TensorProduct<Dim>::first_derivatives(
    const tnsr::I<T, Dim>& x) const noexcept {
  auto result = make_with_value<tnsr::i<T, Dim>>(x, scale_);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t d = 0; d < Dim; ++d) {
      if (i == d) {
        result.get(i) *= gsl::at(functions_, d)->first_deriv(x.get(d));
      } else {
        result.get(i) *= gsl::at(functions_, d)->operator()(x.get(d));
      }
    }
  }
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::ii<T, Dim> TensorProduct<Dim>::second_derivatives(
    const tnsr::I<T, Dim>& x) const noexcept {
  auto result = make_with_value<tnsr::ii<T, Dim>>(x, scale_);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t d = 0; d < Dim; ++d) {
      if (i == d) {
        result.get(i, i) *= gsl::at(functions_, d)->second_deriv(x.get(d));
      } else {
        result.get(i, i) *= gsl::at(functions_, d)->operator()(x.get(d));
      }
    }
    for (size_t j = i + 1; j < Dim; ++j) {
      for (size_t d = 0; d < Dim; ++d) {
        if (i == d or j == d) {
          result.get(i, j) *= gsl::at(functions_, d)->first_deriv(x.get(d));
        } else {
          result.get(i, j) *= gsl::at(functions_, d)->operator()(x.get(d));
        }
      }
    }
  }
  return result;
}

}  // namespace MathFunctions

/// \cond
template class MathFunctions::TensorProduct<1>;
template class MathFunctions::TensorProduct<2>;
template class MathFunctions::TensorProduct<3>;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                             \
  template Scalar<DTYPE(data)> MathFunctions::TensorProduct<DIM(data)>:: \
  operator()(const tnsr::I<DTYPE(data), DIM(data)>& x) const noexcept;   \
  template tnsr::i<DTYPE(data), DIM(data)>                               \
  MathFunctions::TensorProduct<DIM(data)>::first_derivatives(            \
      const tnsr::I<DTYPE(data), DIM(data)>& x) const noexcept;          \
  template tnsr::ii<DTYPE(data), DIM(data)>                              \
  MathFunctions::TensorProduct<DIM(data)>::second_derivatives(           \
      const tnsr::I<DTYPE(data), DIM(data)>& x) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond
