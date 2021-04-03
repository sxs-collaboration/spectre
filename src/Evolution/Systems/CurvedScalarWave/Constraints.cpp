// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace CurvedScalarWave {
template <size_t SpatialDim>
tnsr::i<DataVector, SpatialDim, Frame::Inertial> one_index_constraint(
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& d_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept {
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
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept {
  if (get_size(get<0>(*constraint)) != get_size(get<0>(phi))) {
    *constraint =
        tnsr::i<DataVector, SpatialDim, Frame::Inertial>(get_size(get<0>(phi)));
  }
  // Declare iterators for d_psi and phi outside the for loop,
  // because they are const but constraint is not
  auto d_psi_it = d_psi.cbegin(), phi_it = phi.cbegin();

  for (auto constraint_it = (*constraint).begin();
       constraint_it != (*constraint).end();
       ++constraint_it, (void)++d_psi_it, (void)++phi_it) {
    *constraint_it = *d_psi_it - *phi_it;
  }
}

template <size_t SpatialDim>
tnsr::ij<DataVector, SpatialDim, Frame::Inertial> two_index_constraint(
    const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>& d_phi) noexcept {
  auto constraint =
      make_with_value<tnsr::ij<DataVector, SpatialDim, Frame::Inertial>>(d_phi,
                                                                         0.);
  two_index_constraint<SpatialDim>(make_not_null(&constraint), d_phi);
  return constraint;
}

template <size_t SpatialDim>
void two_index_constraint(
    const gsl::not_null<tnsr::ij<DataVector, SpatialDim, Frame::Inertial>*>
        constraint,
    const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>& d_phi) noexcept {
  if (get_size(get<0, 0>(*constraint)) != get_size(get<0, 0>(d_phi))) {
    *constraint =
        make_with_value<tnsr::ij<DataVector, SpatialDim, Frame::Inertial>>(
            d_phi, 0.);
  }
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
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&) noexcept;       \
  template void CurvedScalarWave::one_index_constraint(                       \
      const gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>,  \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&) noexcept;       \
  template tnsr::ij<DataVector, DIM(data), Frame::Inertial>                   \
  CurvedScalarWave::two_index_constraint(                                     \
      const tnsr::ij<DataVector, DIM(data), Frame::Inertial>&) noexcept;      \
  template void CurvedScalarWave::two_index_constraint(                       \
      const gsl::not_null<tnsr::ij<DataVector, DIM(data), Frame::Inertial>*>, \
      const tnsr::ij<DataVector, DIM(data), Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
