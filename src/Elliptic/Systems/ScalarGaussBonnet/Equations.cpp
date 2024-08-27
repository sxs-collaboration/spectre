// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/ScalarGaussBonnet/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace sgb {

void curved_fluxes(const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const Scalar<DataVector>& conformal_factor,
                   const tnsr::i<DataVector, 3>& field_gradient) {
  raise_or_lower_index(flux_for_field, field_gradient, inv_conformal_metric);
  const DataVector conformal_factor_power = pow<4>(get(conformal_factor));
  const DataVector shift_term =
      get(dot_product(shift, field_gradient)) / square(get(lapse));

  for (size_t i = 0; i < 3; i++) {
    flux_for_field->get(i) /= conformal_factor_power;
    flux_for_field->get(i) -= shift.get(i) * shift_term;
  }
}

void face_fluxes(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                 const tnsr::II<DataVector, 3>& inv_conformal_metric,
                 const tnsr::I<DataVector, 3>& shift,
                 const Scalar<DataVector>& lapse,
                 const Scalar<DataVector>& conformal_factor,
                 const tnsr::i<DataVector, 3>& face_normal,
                 const Scalar<DataVector>& field) {
  raise_or_lower_index(flux_for_field, face_normal, inv_conformal_metric);
  const DataVector conformal_factor_power = pow<4>(get(conformal_factor));
  // This term is constant unless the domain changes (i.e. through AMR), and
  // so in a given non-linear elliptic solve could be precomputed once at the
  // start as a possible performance optimizaton.
  const DataVector shift_term =
      get(dot_product(shift, face_normal)) / square(get(lapse));

  for (size_t i = 0; i < 3; i++) {
    flux_for_field->get(i) /= conformal_factor_power;
    flux_for_field->get(i) -= shift.get(i) * shift_term;
    flux_for_field->get(i) *= get(field);
  }
}

void add_curved_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_field,
    const tnsr::i<DataVector, 3>& deriv_lapse, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv) {
  get(*source_for_field) -=
      get(dot_product(deriv_lapse, flux_for_field)) / get(lapse);
  get(*source_for_field) -=
      get(dot_product(conformal_christoffel_contracted, flux_for_field));
  get(*source_for_field) -=
      6. * get(dot_product(conformal_factor_deriv, flux_for_field)) /
      get(conformal_factor);
}

void add_GB_terms(gsl::not_null<Scalar<DataVector>*> scalar_tensor_equation,
                  const double eps2, const double eps4,
                  const Scalar<DataVector>& weyl_electric,
                  const Scalar<DataVector>& weyl_magnetic,
                  const Scalar<DataVector>& field) {
  get(*scalar_tensor_equation) -=
      2. * (weyl_electric.get() - weyl_magnetic.get()) *
      (eps2 * field.get() + eps4 * cube(get(field)));
}

void add_linearized_GB_terms(
    gsl::not_null<Scalar<DataVector>*> linearized_scalar_tensor_equation,
    const double eps2, const double eps4,
    const Scalar<DataVector>& weyl_electric,
    const Scalar<DataVector>& weyl_magnetic, const Scalar<DataVector>& field,
    const Scalar<DataVector>& field_correction) {
  get(*linearized_scalar_tensor_equation) -=
      2. * (weyl_electric.get() - weyl_magnetic.get()) *
      (eps2 * field_correction.get() +
       3. * eps4 * square(get(field)) * field_correction.get());
}

void Fluxes::apply(const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const Scalar<DataVector>& conformal_factor,
                   const Scalar<DataVector>& /*field*/,
                   const tnsr::i<DataVector, 3>& field_gradient) {
  curved_fluxes(flux_for_field, inv_conformal_metric, shift, lapse,
                conformal_factor, field_gradient);
}

void Fluxes::apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const Scalar<DataVector>& conformal_factor,
                   const tnsr::i<DataVector, 3>& face_normal,
                   const tnsr::I<DataVector, 3>& /*face_normal_vector*/,
                   const Scalar<DataVector>& field) {
  face_fluxes(flux_for_field, inv_conformal_metric, shift, lapse,
              conformal_factor, face_normal, field);
}

void Sources::apply(
    gsl::not_null<Scalar<DataVector>*> equation_for_field,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::i<DataVector, 3>& deriv_lapse, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv, const double& eps2,
    const double& eps4, const Scalar<DataVector>& weyl_electric,
    const Scalar<DataVector>& weyl_magnetic, const Scalar<DataVector>& field,
    const tnsr::I<DataVector, 3>& field_flux) {
  add_curved_sources(equation_for_field, conformal_christoffel_contracted,
                     field_flux, deriv_lapse, lapse, conformal_factor,
                     conformal_factor_deriv);
  add_GB_terms(equation_for_field, eps2, eps4, weyl_electric, weyl_magnetic,
               field);
}

void LinearizedSources::apply(
    gsl::not_null<Scalar<DataVector>*> linearized_equation_for_field,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::i<DataVector, 3>& deriv_lapse, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const Scalar<DataVector>& field, const double& eps2, const double& eps4,
    const Scalar<DataVector>& weyl_electric,
    const Scalar<DataVector>& weyl_magnetic,
    const Scalar<DataVector>& field_correction,
    const tnsr::I<DataVector, 3>& field_flux_correction) {
  add_curved_sources(linearized_equation_for_field,
                     conformal_christoffel_contracted, field_flux_correction,
                     deriv_lapse, lapse, conformal_factor,
                     conformal_factor_deriv);
  add_linearized_GB_terms(linearized_equation_for_field, eps2, eps4,
                          weyl_electric, weyl_magnetic, field,
                          field_correction);
}

}  // namespace sgb
