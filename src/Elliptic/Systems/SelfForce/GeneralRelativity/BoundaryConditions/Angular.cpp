// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/Angular.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace GrSelfForce::BoundaryConditions {

Angular::Angular(const int m_mode_number) : m_mode_number_(m_mode_number) {}

void Angular::apply(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field,
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> n_dot_field_gradient,
    const GradTensorType& /*deriv_field*/) const {
  // Ordering of numbered variables in comments below:
  // tt, tr, ttheta, tphi, rr, rtheta, rphi, theta theta, theta phi, phi phi
  if (m_mode_number_ == 0) {
    ERROR("Not yet implemented for m=0.");
  } else if (m_mode_number_ == 1) {
    // Dirichlet for components 0, 1, 4, 7, 8, 9
    // Neumann for components 2, 3, 5, 6
    get<0, 0>(*field) = 0.;
    get<0, 1>(*field) = 0.;
    get<0, 2>(*n_dot_field_gradient) = 0.;
    get<0, 3>(*n_dot_field_gradient) = 0.;
    get<1, 1>(*field) = 0.;
    get<1, 2>(*n_dot_field_gradient) = 0.;
    get<1, 3>(*n_dot_field_gradient) = 0.;
    get<2, 2>(*field) = 0.;
    get<2, 3>(*field) = 0.;
    get<3, 3>(*field) = 0.;
  } else if (m_mode_number_ == 2) {
    // Dirichlet for components 0-6
    // Neumann for components 7-9
    get<0, 0>(*field) = 0.;
    get<0, 1>(*field) = 0.;
    get<0, 2>(*field) = 0.;
    get<0, 3>(*field) = 0.;
    get<1, 1>(*field) = 0.;
    get<1, 2>(*field) = 0.;
    get<1, 3>(*field) = 0.;
    get<2, 2>(*n_dot_field_gradient) = 0.;
    get<2, 3>(*n_dot_field_gradient) = 0.;
    get<3, 3>(*n_dot_field_gradient) = 0.;
  } else {
    // All Dirichlet
    for (size_t i = 0; i < field->size(); ++i) {
      (*field)[i] = 0.;
    }
  }
}

void Angular::apply_linearized(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field_correction,
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*>
        n_dot_field_correction_gradient,
    const GradTensorType& deriv_field_correction) const {
  apply(field_correction, n_dot_field_correction_gradient,
        deriv_field_correction);
}

void Angular::pup(PUP::er& p) {
  Base::pup(p);
  p | m_mode_number_;
}

bool operator==(const Angular& lhs, const Angular& rhs) {
  return lhs.m_mode_number_ == rhs.m_mode_number_;
}

bool operator!=(const Angular& lhs, const Angular& rhs) {
  return not(lhs == rhs);
}

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID Angular::my_PUP_ID = 0;  // NOLINT
#endif                                     // SPECTRE_USE_CHARM

}  // namespace GrSelfForce::BoundaryConditions
