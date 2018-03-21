// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/Evolution/Systems/CurvedScalarWave/TestHelpers.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

template <typename DataType>
Scalar<DataType> make_pi(const DataType& used_for_size) {
  return make_with_value<Scalar<DataType>>(used_for_size, -1.6);
}

template <size_t Dim, typename DataType>
tnsr::i<DataType, Dim> make_phi(const DataType& used_for_size) {
  auto phi = make_with_value<tnsr::i<DataType, Dim>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    phi.get(i) = make_with_value<DataType>(used_for_size, -3. * i + 0.5);
  }
  return phi;
}

template <size_t Dim, typename DataType>
tnsr::i<DataType, Dim> make_d_psi(const DataType& used_for_size) {
  auto d_psi = make_with_value<tnsr::i<DataType, Dim>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    d_psi.get(i) = make_with_value<DataType>(used_for_size, 2. * i + 0.5);
  }
  return d_psi;
}

template <size_t Dim, typename DataType>
tnsr::i<DataType, Dim> make_d_pi(const DataType& used_for_size) {
  auto d_pi = make_with_value<tnsr::i<DataType, Dim>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    d_pi.get(i) = make_with_value<DataType>(used_for_size, 2.5 * i + 0.5);
  }
  return d_pi;
}

template <size_t Dim, typename DataType>
tnsr::ij<DataType, Dim> make_d_phi(const DataType& used_for_size) {
  auto d_phi = make_with_value<tnsr::ij<DataType, Dim>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      d_phi.get(i, j) =
          make_with_value<DataType>(used_for_size, 5. * (i - 0.5) + 0.5 + j);
    }
  }
  return d_phi;
}

template <typename DataType>
Scalar<DataType> make_constraint_gamma1(const DataType& used_for_size) {
  return make_with_value<Scalar<DataType>>(used_for_size, 5.8);
}

template <typename DataType>
Scalar<DataType> make_constraint_gamma2(const DataType& used_for_size) {
  return make_with_value<Scalar<DataType>>(used_for_size, -4.3);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                           \
  template tnsr::i<DataVector, DIM(data)> make_phi(    \
      const DataVector& used_for_size);                \
  template tnsr::i<DataVector, DIM(data)> make_d_psi(  \
      const DataVector& used_for_size);                \
  template tnsr::i<DataVector, DIM(data)> make_d_pi(   \
      const DataVector& used_for_size);                \
  template tnsr::ij<DataVector, DIM(data)> make_d_phi( \
      const DataVector& used_for_size);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

template Scalar<DataVector> make_pi(const DataVector& used_for_size);
template Scalar<DataVector> make_constraint_gamma1(
    const DataVector& used_for_size);
template Scalar<DataVector> make_constraint_gamma2(
    const DataVector& used_for_size);

#undef DIM
#undef DTYPE
#undef INSTANTIATE
