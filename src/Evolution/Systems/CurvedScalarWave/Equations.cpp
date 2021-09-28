// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"

#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"      // IWYU pragma: keep
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

template <typename X, typename Symm, typename IndexList>
class Tensor;

namespace CurvedScalarWave {
template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> phi_normal_dot_flux,
    const gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
    const Scalar<DataVector>& pi, const tnsr::i<DataVector, Dim>& phi,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& gamma1,
    const Scalar<DataVector>& gamma2, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim>& shift,
    const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
    const tnsr::i<DataVector, Dim>& interface_unit_normal) {
  const auto shift_dot_normal = get(dot_product(shift, interface_unit_normal));
  const auto normal_dot_phi = [&]() {
    auto normal_dot_phi_ = make_with_value<Scalar<DataVector>>(psi, 0.);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        get(normal_dot_phi_) += inverse_spatial_metric.get(i, j) *
                                interface_unit_normal.get(j) * phi.get(i);
      }
    }
    return normal_dot_phi_;
  }();

  psi_normal_dot_flux->get() =
      -(1. + get(gamma1)) * shift_dot_normal * get(psi);

  pi_normal_dot_flux->get() =
      -shift_dot_normal * get(pi) + get(lapse) * get(normal_dot_phi) -
      get(gamma1) * get(gamma2) * shift_dot_normal * get(psi);

  for (size_t i = 0; i < Dim; ++i) {
    phi_normal_dot_flux->get(i) =
        get(lapse) * (interface_unit_normal.get(i) * get(pi) -
                      get(gamma2) * interface_unit_normal.get(i) * get(psi)) -
        shift_dot_normal * phi.get(i);
  }
}
}  // namespace CurvedScalarWave
// Generate explicit instantiations of partial_derivatives function as well as
// all other functions in Equations.cpp

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) \
  template struct CurvedScalarWave::ComputeNormalDotFluxes<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
