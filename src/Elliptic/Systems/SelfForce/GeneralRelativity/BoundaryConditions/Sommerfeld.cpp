// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/Sommerfeld.hpp"

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace GrSelfForce::BoundaryConditions {

Sommerfeld::Sommerfeld(const double black_hole_mass,
                       const double black_hole_spin,
                       const double orbital_radius, const int m_mode_number)
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      orbital_radius_(orbital_radius),
      m_mode_number_(m_mode_number) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Sommerfeld::get_clone() const {
  return std::make_unique<Sommerfeld>(*this);
}

void Sommerfeld::apply(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field,
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> n_dot_field_gradient,
    const GradTensorType& /*deriv_field*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_0 = orbital_radius_;
  const double omega = 1. / (a + sqrt(cube(r_0) / M));
  for (size_t i = 0; i < field->size(); ++i) {
    (*n_dot_field_gradient)[i] =
        std::complex<double>(0.0, m_mode_number_ * omega) * (*field)[i];
  }
}

void Sommerfeld::apply_linearized(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field_correction,
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*>
        n_dot_field_correction_gradient,
    const GradTensorType& deriv_field_correction) const {
  apply(field_correction, n_dot_field_correction_gradient,
        deriv_field_correction);
}

void Sommerfeld::pup(PUP::er& p) {
  p | black_hole_mass_;
  p | black_hole_spin_;
  p | orbital_radius_;
  p | m_mode_number_;
}

bool operator==(const Sommerfeld& lhs, const Sommerfeld& rhs) {
  return lhs.black_hole_mass_ == rhs.black_hole_mass_ and
         lhs.black_hole_spin_ == rhs.black_hole_spin_ and
         lhs.orbital_radius_ == rhs.orbital_radius_ and
         lhs.m_mode_number_ == rhs.m_mode_number_;
}

bool operator!=(const Sommerfeld& lhs, const Sommerfeld& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID Sommerfeld::my_PUP_ID = 0;  // NOLINT

}  // namespace GrSelfForce::BoundaryConditions
