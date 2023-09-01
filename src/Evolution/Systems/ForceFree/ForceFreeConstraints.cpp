// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/ForceFreeConstraints.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {

void tilde_e_or_b_squared(
    const gsl::not_null<Scalar<DataVector>*> tilde_e_or_b_squared,
    const tnsr::I<DataVector, 3>& densitized_vector,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  dot_product(tilde_e_or_b_squared, densitized_vector, densitized_vector,
              spatial_metric);
}

void tilde_e_dot_tilde_b(
    const gsl::not_null<Scalar<DataVector>*> tilde_e_dot_tilde_b,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  dot_product(tilde_e_dot_tilde_b, tilde_e, tilde_b, spatial_metric);
}

void electric_field_dot_magnetic_field(
    const gsl::not_null<Scalar<DataVector>*> electric_field_dot_magnetic_field,
    const tnsr::I<DataVector, 3>& tilde_e,
    const tnsr::I<DataVector, 3>& tilde_b,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  tilde_e_dot_tilde_b(electric_field_dot_magnetic_field, tilde_e, tilde_b,
                      spatial_metric);
  get(*electric_field_dot_magnetic_field) /=
      square(get(sqrt_det_spatial_metric));
}

void magnetic_dominance_violation(
    const gsl::not_null<Scalar<DataVector>*> magnetic_dominance_violation,
    const Scalar<DataVector>& tilde_e_squared,
    const Scalar<DataVector>& tilde_b_squared,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  get(*magnetic_dominance_violation) =
      max(get(tilde_e_squared) - get(tilde_b_squared), 0.0);

  get(*magnetic_dominance_violation) /= square(get(sqrt_det_spatial_metric));
}

}  // namespace ForceFree
