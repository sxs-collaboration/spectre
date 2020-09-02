// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic {
/// \cond
template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<tnsr::aa<DataVector, Dim>*>
        spacetime_metric_normal_dot_flux,
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::iaa<DataVector, Dim>*> phi_normal_dot_flux,
    const tnsr::aa<DataVector, Dim>& spacetime_metric) noexcept {
  destructive_resize_components(pi_normal_dot_flux,
                                get<0, 0>(spacetime_metric).size());
  destructive_resize_components(phi_normal_dot_flux,
                                get<0, 0>(spacetime_metric).size());
  destructive_resize_components(spacetime_metric_normal_dot_flux,
                                get<0, 0>(spacetime_metric).size());
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
/// \endcond
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) \
  template struct GeneralizedHarmonic::ComputeNormalDotFluxes<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
/// \endcond
