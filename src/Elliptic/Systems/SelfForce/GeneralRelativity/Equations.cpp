// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Utilities/Algorithm.hpp"

namespace GrSelfForce {

void fluxes(const gsl::not_null<FluxTensorType*> flux,
            const tnsr::I<ComplexDataVector, 2>& alpha,
            const GradTensorType& field_gradient) {
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      flux->get(0, a, b) = get<0>(alpha) * field_gradient.get(0, a, b);
      flux->get(1, a, b) = get<1>(alpha) * field_gradient.get(1, a, b);
    }
  }
}

void fluxes_on_face(const gsl::not_null<FluxTensorType*> flux,
                    const tnsr::I<ComplexDataVector, 2>& alpha,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const tnsr::aa<ComplexDataVector, 3>& field) {
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      flux->get(0, a, b) =
          get<0>(alpha) * get<0>(face_normal_vector) * field.get(a, b);
      flux->get(1, a, b) =
          get<1>(alpha) * get<1>(face_normal_vector) * field.get(a, b);
    }
  }
}

void add_sources(const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> source,
                 const tnsr::aaBB<ComplexDataVector, 3>& beta,
                 const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar,
                 const tnsr::aaBB<ComplexDataVector, 3>& gamma_theta,
                 const tnsr::aa<ComplexDataVector, 3>& field,
                 const GradTensorType& field_gradient) {
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      for (size_t c = 0; c < 4; ++c) {
        for (size_t d = 0; d <= c; ++d) {
          source->get(a, b) +=
              beta.get(a, b, c, d) * field.get(c, d) +
              gamma_rstar.get(a, b, c, d) * field_gradient.get(0, c, d) +
              gamma_theta.get(a, b, c, d) * field_gradient.get(1, c, d);
        }
      }
    }
  }
}

void Fluxes::apply(const gsl::not_null<FluxTensorType*> flux,
                   const tnsr::I<ComplexDataVector, 2>& alpha,
                   const tnsr::aa<ComplexDataVector, 3>& /*field*/,
                   const GradTensorType& field_gradient) {
  fluxes(flux, alpha, field_gradient);
}

void Fluxes::apply(const gsl::not_null<FluxTensorType*> flux,
                   const tnsr::I<ComplexDataVector, 2>& alpha,
                   const tnsr::i<DataVector, 2>& /*face_normal*/,
                   const tnsr::I<DataVector, 2>& face_normal_vector,
                   const tnsr::aa<ComplexDataVector, 3>& field) {
  fluxes_on_face(flux, alpha, face_normal_vector, field);
}

void Sources::apply(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> scalar_equation,
    const tnsr::aaBB<ComplexDataVector, 3>& beta,
    const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar,
    const tnsr::aaBB<ComplexDataVector, 3>& gamma_theta,
    const tnsr::aa<ComplexDataVector, 3>& field,
    const GradTensorType& field_gradient, const FluxTensorType& /*flux*/) {
  add_sources(scalar_equation, beta, gamma_rstar, gamma_theta, field,
              field_gradient);
}

void ModifyBoundaryData::apply(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field,
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> n_dot_flux,
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
  for (size_t i = 0; i < singular_field.size(); ++i) {
    (*field)[i] += sign * singular_field[i];
    (*n_dot_flux)[i] -= sign * singular_field_n_dot_flux[i];
  }
}

void ModifyBoundaryData::apply_linearized(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field_remote,
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*>
        n_dot_flux_remote,
    const tnsr::aa<ComplexDataVector, 3>& field_local,
    const tnsr::aa<ComplexDataVector, 3>& /*n_dot_flux_local*/,
    const DirectionalId<Dim>& mortar_id, const Element<Dim>& element,
    const std::vector<size_t>& null_slicing_blocks,
    const elliptic::analytic_data::Background& background) {
  if (alg::found(null_slicing_blocks, element.id().block_id()) ==
      alg::found(null_slicing_blocks, mortar_id.id().block_id())) {
    // Both elements use the same slicing. Nothing to do.
    return;
  }
  // Apply the jump in the flux across the boundary to handle
  // vtu-slicing. The signs are all the same (on both sides of the boundary and
  // at both transition points).
  const auto& circular_orbit =
      dynamic_cast<const GrSelfForce::AnalyticData::CircularOrbit&>(background);
  const double omega = circular_orbit.omega();
  const double m_mode_number = circular_orbit.m_mode_number();
  for (size_t j = 0; j < n_dot_flux_remote->size(); ++j) {
    (*n_dot_flux_remote)[j] -=
        std::complex<double>(0.0, m_mode_number * omega) *
        ((field_local)[j] + (*field_remote)[j]) * 0.5;
  }
}

}  // namespace GrSelfForce
