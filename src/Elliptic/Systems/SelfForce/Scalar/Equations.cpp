// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/Equations.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/CircularOrbit.hpp"
#include "Utilities/Algorithm.hpp"

namespace ScalarSelfForce {

void fluxes(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
            const Scalar<ComplexDataVector>& alpha,
            const tnsr::i<ComplexDataVector, 2>& field_gradient) {
  get<0>(*flux) = get<0>(field_gradient);
  get<1>(*flux) = get(alpha) * get<1>(field_gradient);
}

void fluxes_on_face(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const Scalar<ComplexDataVector>& field) {
  get<0>(*flux) = get<0>(face_normal_vector) * get(field);
  get<1>(*flux) = get(alpha) * get<1>(face_normal_vector) * get(field);
}

void add_sources(const gsl::not_null<Scalar<ComplexDataVector>*> source,
                 const Scalar<ComplexDataVector>& beta,
                 const tnsr::i<ComplexDataVector, 2>& gamma,
                 const Scalar<ComplexDataVector>& field,
                 const tnsr::I<ComplexDataVector, 2>& flux) {
  get(*source) += get(beta) * get(field) + get<0>(gamma) * get<0>(flux) +
                  get<1>(gamma) * get<1>(flux);
}

void Fluxes::apply(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                   const Scalar<ComplexDataVector>& alpha,
                   const Scalar<ComplexDataVector>& /*field*/,
                   const tnsr::i<ComplexDataVector, 2>& field_gradient) {
  fluxes(flux, alpha, field_gradient);
}

void Fluxes::apply(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                   const Scalar<ComplexDataVector>& alpha,
                   const tnsr::i<DataVector, 2>& /*face_normal*/,
                   const tnsr::I<DataVector, 2>& face_normal_vector,
                   const Scalar<ComplexDataVector>& field) {
  fluxes_on_face(flux, alpha, face_normal_vector, field);
}

void Sources::apply(
    const gsl::not_null<Scalar<ComplexDataVector>*> scalar_equation,
    const Scalar<ComplexDataVector>& beta,
    const tnsr::i<ComplexDataVector, 2>& gamma,
    const Scalar<ComplexDataVector>& field,
    const tnsr::i<ComplexDataVector, 2>& /*field_gradient*/,
    const tnsr::I<ComplexDataVector, 2>& flux) {
  add_sources(scalar_equation, beta, gamma, field, flux);
}

void ModifyBoundaryData::apply(
    const gsl::not_null<Scalar<ComplexDataVector>*> field,
    const gsl::not_null<Scalar<ComplexDataVector>*> n_dot_flux,
    const DirectionalId<Dim>& mortar_id, const bool field_is_regularized,
    const DirectionalIdMap<Dim, bool>& neighbors_field_is_regularized,
    const DirectionalIdMap<Dim, typename singular_vars_on_mortars_tag::type>&
        singular_vars_on_mortars) {
  if (field_is_regularized == neighbors_field_is_regularized.at(mortar_id)) {
    // Both elements solve for the same field. Nothing to do.
    return;
  }
  // Subtract the singular field on the regularized side, and add it on the
  // other side
  const double sign = field_is_regularized ? -1. : 1.;
  const auto& singular_field =
      get<Tags::SingularField>(singular_vars_on_mortars.at(mortar_id));
  const auto& singular_field_n_dot_flux =
      get<::Tags::NormalDotFlux<Tags::SingularField>>(
          singular_vars_on_mortars.at(mortar_id));
  get(*field) += sign * get(singular_field);
  get(*n_dot_flux) -= sign * get(singular_field_n_dot_flux);
}

void ModifyBoundaryData::apply_linearized(
    const gsl::not_null<Scalar<ComplexDataVector>*> /*field_remote*/,
    const gsl::not_null<
        Scalar<ComplexDataVector>*> n_dot_field_gradient_remote,
    const gsl::not_null<Scalar<ComplexDataVector>*> /*field_local*/,
    const gsl::not_null<
        Scalar<ComplexDataVector>*> /*n_dot_field_gradient_local*/,
    const Scalar<ComplexDataVector>& avg_field,
    const DirectionalId<Dim>& mortar_id, const Element<Dim>& element,
    const std::vector<size_t>& null_slicing_blocks,
    const elliptic::analytic_data::Background& background) {
  if (alg::found(null_slicing_blocks, element.id().block_id()) ==
      alg::found(null_slicing_blocks, mortar_id.id().block_id())) {
    // Both elements use the same slicing. Nothing to do.
    return;
  }
  // Apply the jump in the field gradient across the boundary to handle
  // vtu-slicing. The signs are all the same (on both sides of the boundary and
  // at both transition points).
  const auto& circular_orbit =
      dynamic_cast<const ScalarSelfForce::AnalyticData::CircularOrbit&>(
          background);
  const double omega = circular_orbit.omega();
  const double m_mode_number = circular_orbit.m_mode_number();
  get(*n_dot_field_gradient_remote) -=
      std::complex<double>(0.0, m_mode_number * omega) * get(avg_field);
}

}  // namespace ScalarSelfForce
