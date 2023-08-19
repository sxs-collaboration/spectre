// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/ElectromagneticVariables.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree {

void em_field_from_evolved_fields(
    const gsl::not_null<tnsr::I<DataVector, 3>*> vector,
    const tnsr::I<DataVector, 3>& densitized_vector,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  get<0>(*vector) = get<0>(densitized_vector) / get(sqrt_det_spatial_metric);
  get<1>(*vector) = get<1>(densitized_vector) / get(sqrt_det_spatial_metric);
  get<2>(*vector) = get<2>(densitized_vector) / get(sqrt_det_spatial_metric);
}

void charge_density_from_tilde_q(
    const gsl::not_null<Scalar<DataVector>*> charge_density,
    const Scalar<DataVector>& tilde_q,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  get(*charge_density) = get(tilde_q) / get(sqrt_det_spatial_metric);
}

void electric_current_density_from_tilde_j(
    const gsl::not_null<tnsr::I<DataVector, 3>*> electric_current_density,
    const tnsr::I<DataVector, 3>& tilde_j,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& lapse) {
  em_field_from_evolved_fields(electric_current_density, tilde_j,
                               sqrt_det_spatial_metric);
  get<0>(*electric_current_density) =
      get<0>(*electric_current_density) / get(lapse);
  get<1>(*electric_current_density) =
      get<1>(*electric_current_density) / get(lapse);
  get<2>(*electric_current_density) =
      get<2>(*electric_current_density) / get(lapse);
}

}  // namespace ForceFree
