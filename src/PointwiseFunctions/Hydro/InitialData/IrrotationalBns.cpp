// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/InitialData/IrrotationalBns.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"

#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"

#include "Utilities/Gsl.hpp"

namespace hydro::initial_data {

void rotational_shift(
    const gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_rotational_killing_vector) {
  ::tenex::evaluate<ti::I>(
      result, shift(ti::I) + spatial_rotational_killing_vector(ti::I));
}

tnsr::I<DataVector, 3> rotational_shift(
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_rotational_killing_vector) {
  tnsr::I<DataVector, 3> buffer{};
  rotational_shift(make_not_null(&buffer), shift,
                   spatial_rotational_killing_vector);
  return buffer;
}

void rotational_shift_stress(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& rotational_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric) {
  ::tenex::evaluate<ti::I, ti::j>(
      result, rotational_shift(ti::I) * spatial_metric(ti::k, ti::j) *
                  rotational_shift(ti::K) / square(lapse()));
}

tnsr::Ij<DataVector, 3> rotational_shift_stress(
    const tnsr::I<DataVector, 3>& rotational_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric) {
  tnsr::Ij<DataVector, 3> buffer{};
  rotational_shift_stress(make_not_null(&buffer), rotational_shift, lapse,
                          spatial_metric);
  return buffer;
}

void derivative_rotational_shift_over_lapse(
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& rotational_shift,
    const tnsr::iJ<DataVector, 3>& deriv_of_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& deriv_of_lapse,
    const tnsr::iJ<DataVector, 3>& deriv_of_spatial_rotational_killing_vector) {
  ::tenex::evaluate<ti::i, ti::J>(
      result,
      (deriv_of_shift(ti::i, ti::J) +
       deriv_of_spatial_rotational_killing_vector(ti::i, ti::J)) /
              lapse() -
          rotational_shift(ti::J) * deriv_of_lapse(ti::i) / square(lapse()));
}

tnsr::iJ<DataVector, 3> derivative_rotational_shift_over_lapse(
    const tnsr::I<DataVector, 3>& rotational_shift,
    const tnsr::iJ<DataVector, 3>& deriv_of_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& deriv_of_lapse,
    const tnsr::iJ<DataVector, 3>& deriv_of_spatial_rotational_killing_vector) {
  tnsr::iJ<DataVector, 3> buffer{};
  derivative_rotational_shift_over_lapse(
      make_not_null(&buffer), rotational_shift, deriv_of_shift, lapse,
      deriv_of_lapse, deriv_of_spatial_rotational_killing_vector);
  return buffer;
}

void divergence_rotational_shift_stress(
    const gsl::not_null<tnsr::i<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& rotational_shift,
    const tnsr::iJ<DataVector, 3>& derivative_rotational_shift_over_lapse,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric) {
  // This has symmetry properties that should be used
  ::tenex::evaluate<ti::k>(
      result, rotational_shift(ti::J) *
                      derivative_rotational_shift_over_lapse(ti::j, ti::I) *
                      spatial_metric(ti::i, ti::k) / lapse() +
                  rotational_shift(ti::I) * spatial_metric(ti::i, ti::k) *
                      derivative_rotational_shift_over_lapse(ti::j, ti::J) /
                      lapse());
}

tnsr::i<DataVector, 3> divergence_rotational_shift_stress(
    const tnsr::I<DataVector, 3>& rotational_shift,
    const tnsr::iJ<DataVector, 3>& derivative_rotational_shift_over_lapse,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric) {
  tnsr::i<DataVector, 3> buffer{};
  divergence_rotational_shift_stress(make_not_null(&buffer), rotational_shift,
                                     derivative_rotational_shift_over_lapse,
                                     lapse, spatial_metric);
  return buffer;
}

void enthalpy_density_squared(
    const gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::I<DataVector, 3>& rotational_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& velocity_potential_gradient,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const double euler_enthalpy_constant) {
  return tenex::evaluate<>(
      result,
      square((euler_enthalpy_constant +
              rotational_shift(ti::I) * velocity_potential_gradient(ti::i)) /
             lapse()) -
          velocity_potential_gradient(ti::i) *
              velocity_potential_gradient(ti::j) *
              inverse_spatial_metric(ti::I, ti::J));
}

Scalar<DataVector> enthalpy_density_squared(
    const tnsr::I<DataVector, 3>& rotational_shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& velocity_potential_gradient,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const double euler_enthalpy_constant) {
  Scalar<DataVector> buffer{};
  enthalpy_density_squared(make_not_null(&buffer), rotational_shift, lapse,
                           velocity_potential_gradient, inverse_spatial_metric,
                           euler_enthalpy_constant);
  return buffer;
}

void spatial_rotational_killing_vector(
    const gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& local_angular_velocity_around_z,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  // Cross product involves volume element in arbitrary coordinates
  result->get(0) = -get(sqrt_det_spatial_metric) * get<1>(x) *
                   get(local_angular_velocity_around_z);
  result->get(1) = get(sqrt_det_spatial_metric) * get<0>(x) *
                   get(local_angular_velocity_around_z);
  result->get(2) = make_with_value<DataVector>(sqrt_det_spatial_metric, 0.0);
}

tnsr::I<DataVector, 3> spatial_rotational_killing_vector(
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& local_angular_velocity_around_z,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  tnsr::I<DataVector, 3> buffer{};
  spatial_rotational_killing_vector(make_not_null(&buffer), x,
                                    local_angular_velocity_around_z,
                                    sqrt_det_spatial_metric);
  return buffer;
}

void derivative_spatial_rotational_killing_vector(
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& /*local_angular_velocity_around_z*/,
    const Scalar<DataVector>& /*sqrt_det_spatial_metric*/) {
  *result = make_with_value<tnsr::iJ<DataVector, 3>>(get<0>(x), 0.0);
}

tnsr::iJ<DataVector, 3> derivative_spatial_rotational_killing_vector(
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& local_angular_velocity_around_z,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  tnsr::iJ<DataVector, 3> buffer{};
  derivative_spatial_rotational_killing_vector(make_not_null(&buffer), x,
                                               local_angular_velocity_around_z,
                                               sqrt_det_spatial_metric);
  return buffer;
}
}  // namespace hydro::initial_data
