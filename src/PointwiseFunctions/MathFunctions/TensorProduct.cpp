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
    double scale,
    std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, Dim>&&
        functions)
    : scale_(scale), functions_(std::move(functions)) {}

template <size_t Dim>
TensorProduct<Dim>::TensorProduct(const TensorProduct<Dim>& other)
    : scale_(other.scale_) {
  for (size_t i = 0; i < Dim; ++i) {
    functions_.at(i) = other.functions_.at(i)->get_clone();
  }
}

template <size_t Dim>
TensorProduct<Dim>& TensorProduct<Dim>::operator=(
    const TensorProduct<Dim>& other) {
  scale_ = other.scale_;
  for (size_t i = 0; i < Dim; ++i) {
    functions_.at(i) = other.functions_.at(i)->get_clone();
  }
  return *this;
}

template <size_t Dim>
template <typename T>
Scalar<T> TensorProduct<Dim>::operator()(const tnsr::I<T, Dim>& x) const {
  auto result = make_with_value<Scalar<T>>(x, scale_);
  for (size_t d = 0; d < Dim; ++d) {
    result.get() *= gsl::at(functions_, d)->operator()(x.get(d));
  }
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::i<T, Dim> TensorProduct<Dim>::first_derivatives(
    const tnsr::I<T, Dim>& x) const {
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
    const tnsr::I<T, Dim>& x) const {
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
template <size_t Dim>
bool operator==(const TensorProduct<Dim>& lhs, const TensorProduct<Dim>& rhs) {
  bool are_equal = lhs.scale_ == rhs.scale_;
  for (size_t i = 0; i < Dim; ++i) {
    are_equal = are_equal and *lhs.functions_.at(i) == *rhs.functions_.at(i);
  }
  return are_equal;
}
template <size_t Dim>
bool operator!=(const TensorProduct<Dim>& lhs, const TensorProduct<Dim>& rhs) {
  return !(lhs == rhs);
}

}  // namespace MathFunctions

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                   \
  template Scalar<DTYPE(data)>                                 \
  MathFunctions::TensorProduct<DIM(data)>::operator()(         \
      const tnsr::I<DTYPE(data), DIM(data)>& x) const;         \
  template tnsr::i<DTYPE(data), DIM(data)>                     \
  MathFunctions::TensorProduct<DIM(data)>::first_derivatives(  \
      const tnsr::I<DTYPE(data), DIM(data)>& x) const;         \
  template tnsr::ii<DTYPE(data), DIM(data)>                    \
  MathFunctions::TensorProduct<DIM(data)>::second_derivatives( \
      const tnsr::I<DTYPE(data), DIM(data)>& x) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                               \
  template class MathFunctions::TensorProduct<DIM(data)>;  \
  template bool MathFunctions::operator==(                 \
      const MathFunctions::TensorProduct<DIM(data)>& lhs,  \
      const MathFunctions::TensorProduct<DIM(data)>& rhs); \
  template bool MathFunctions::operator!=(                 \
      const MathFunctions::TensorProduct<DIM(data)>& lhs,  \
      const MathFunctions::TensorProduct<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
