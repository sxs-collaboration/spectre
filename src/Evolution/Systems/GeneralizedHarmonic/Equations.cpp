// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace gh {
template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<tnsr::aa<DataVector, Dim>*>
        spacetime_metric_normal_dot_flux,
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::iaa<DataVector, Dim>*> phi_normal_dot_flux,
    const tnsr::aa<DataVector, Dim>& spacetime_metric) {
  set_number_of_grid_points(pi_normal_dot_flux, spacetime_metric);
  set_number_of_grid_points(phi_normal_dot_flux, spacetime_metric);
  set_number_of_grid_points(spacetime_metric_normal_dot_flux, spacetime_metric);
  for (size_t storage_index = 0; storage_index < pi_normal_dot_flux->size();
       ++storage_index) {
    (*pi_normal_dot_flux)[storage_index] = 0.0;
    (*spacetime_metric_normal_dot_flux)[storage_index] = 0.0;
  }

  for (size_t storage_index = 0; storage_index < phi_normal_dot_flux->size();
       ++storage_index) {
    (*phi_normal_dot_flux)[storage_index] = 0.0;
  }
}
}  // namespace gh

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) \
  template struct gh::ComputeNormalDotFluxes<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
