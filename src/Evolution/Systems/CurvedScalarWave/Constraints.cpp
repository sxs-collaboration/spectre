// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave {
template <size_t SpatialDim>
tnsr::i<DataVector, SpatialDim, Frame::Inertial> one_index_constraint(
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& d_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) {
  tnsr::i<DataVector, SpatialDim, Frame::Inertial> constraint(
      get_size(get<0>(phi)));
  one_index_constraint<SpatialDim>(make_not_null(&constraint), d_psi, phi);
  return constraint;
}

template <size_t SpatialDim>
void one_index_constraint(
    const gsl::not_null<tnsr::i<DataVector, SpatialDim, Frame::Inertial>*>
        constraint,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& d_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) {
  destructive_resize_components(constraint, get<0>(phi).size());
  tenex::evaluate<ti::i>(constraint, d_psi(ti::i) - phi(ti::i));
}

template <size_t SpatialDim>
tnsr::ij<DataVector, SpatialDim, Frame::Inertial> two_index_constraint(
    const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>& d_phi) {
  tnsr::ij<DataVector, SpatialDim, Frame::Inertial> constraint{};
  two_index_constraint<SpatialDim>(make_not_null(&constraint), d_phi);
  return constraint;
}

template <size_t SpatialDim>
void two_index_constraint(
    const gsl::not_null<tnsr::ij<DataVector, SpatialDim, Frame::Inertial>*>
        constraint,
    const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>& d_phi) {
  destructive_resize_components(constraint, get<0, 0>(d_phi).size());
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      constraint->get(i, j) = d_phi.get(i, j) - d_phi.get(j, i);
    }
  }
}
}  // namespace CurvedScalarWave

// Explicit Instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::i<DataVector, DIM(data), Frame::Inertial>                    \
  CurvedScalarWave::one_index_constraint(                                     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&);                \
  template void CurvedScalarWave::one_index_constraint(                       \
      const gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>,  \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&);                \
  template tnsr::ij<DataVector, DIM(data), Frame::Inertial>                   \
  CurvedScalarWave::two_index_constraint(                                     \
      const tnsr::ij<DataVector, DIM(data), Frame::Inertial>&);               \
  template void CurvedScalarWave::two_index_constraint(                       \
      const gsl::not_null<tnsr::ij<DataVector, DIM(data), Frame::Inertial>*>, \
      const tnsr::ij<DataVector, DIM(data), Frame::Inertial>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
