// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Angular.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarSelfForce::BoundaryConditions {

Angular::Angular(int m_mode_number) : m_mode_number_(m_mode_number) {}

Angular::Angular(CkMigrateMessage* m) : Base(m) {}

void Angular::apply(
    const gsl::not_null<Scalar<ComplexDataVector>*> field,
    const gsl::not_null<Scalar<ComplexDataVector>*> n_dot_field_gradient,
    const tnsr::i<ComplexDataVector, 2>& /*deriv_field*/) const {
  if (m_mode_number_ == 0) {
    get(*n_dot_field_gradient) = 0.;
  } else {
    get(*field) = 0.;
  }
}

void Angular::apply_linearized(
    const gsl::not_null<Scalar<ComplexDataVector>*> field_correction,
    const gsl::not_null<Scalar<ComplexDataVector>*>
        n_dot_field_gradient_correction,
    const tnsr::i<ComplexDataVector, 2>& deriv_field_correction) const {
  apply(field_correction, n_dot_field_gradient_correction,
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

#ifndef __CUDA_ARCH__
PUP::able::PUP_ID Angular::my_PUP_ID = 0;  // NOLINT
#endif                                     // __CUDA_ARCH__

}  // namespace ScalarSelfForce::BoundaryConditions
