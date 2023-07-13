// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave {
template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
    const gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_normal_dot_flux,
    const Scalar<DataVector>& pi) {
  set_number_of_grid_points(psi_normal_dot_flux, pi);
  set_number_of_grid_points(pi_normal_dot_flux, pi);
  set_number_of_grid_points(phi_normal_dot_flux, pi);
  get(*pi_normal_dot_flux) = 0.0;
  get(*psi_normal_dot_flux) = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    phi_normal_dot_flux->get(i) = 0.0;
  }
}
}  // namespace ScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) \
  template class ScalarWave::ComputeNormalDotFluxes<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
