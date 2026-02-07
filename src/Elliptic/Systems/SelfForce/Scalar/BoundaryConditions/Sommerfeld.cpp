// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Sommerfeld.hpp"

#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarSelfForce::BoundaryConditions {

Sommerfeld::Sommerfeld(const double black_hole_mass,
                       const double black_hole_spin,
                       const double orbital_radius, const int m_mode_number,
                       const bool hyperboloidal_slicing, const int order)
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      orbital_radius_(orbital_radius),
      m_mode_number_(m_mode_number),
      hyperboloidal_slicing_(hyperboloidal_slicing),
      order_(order) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Sommerfeld::get_clone() const {
  return std::make_unique<Sommerfeld>(*this);
}

void Sommerfeld::apply(
    const gsl::not_null<Scalar<ComplexDataVector>*> field,
    const gsl::not_null<Scalar<ComplexDataVector>*> n_dot_field_gradient,
    const tnsr::i<ComplexDataVector, 2>& /*deriv_field*/,
    const Scalar<ComplexDataVector>& beta,
    const tnsr::i<ComplexDataVector, 2>& gamma) const {
  if (hyperboloidal_slicing_) {
    if (order_ == 1) {
      get(*n_dot_field_gradient) = 0.;
    } else if (order_ == 2) {
      get(*n_dot_field_gradient) = -get(beta) / get<0>(gamma) * get(*field);
    } else {
      ERROR("Order " << order_
                     << " not implemented for Sommerfeld boundary condition "
                        "with hyperboloidal slicing.");
    }
    return;
  }
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_0 = orbital_radius_;
  const double omega = 1. / (a + sqrt(cube(r_0) / M));
  const double k = m_mode_number_ * omega;
  if (order_ == 1) {
    get(*n_dot_field_gradient) = std::complex<double>(0.0, k) * get(*field);
  } else if (order_ == 2) {
    get(*n_dot_field_gradient) =
        (square(k) - get(beta)) /
        (get<0>(gamma) - std::complex<double>(0.0, 2. * k)) * get(*field);
  } else {
    ERROR("Order " << order_
                   << " not implemented for Sommerfeld boundary condition.");
  }
}

void Sommerfeld::apply_linearized(
    const gsl::not_null<Scalar<ComplexDataVector>*> field_correction,
    const gsl::not_null<Scalar<ComplexDataVector>*>
        n_dot_field_gradient_correction,
    const tnsr::i<ComplexDataVector, 2>& deriv_field_correction,
    const Scalar<ComplexDataVector>& beta,
    const tnsr::i<ComplexDataVector, 2>& gamma) const {
  apply(field_correction, n_dot_field_gradient_correction,
        deriv_field_correction, beta, gamma);
}

void Sommerfeld::pup(PUP::er& p) {
  Base::pup(p);
  p | black_hole_mass_;
  p | black_hole_spin_;
  p | orbital_radius_;
  p | m_mode_number_;
  p | hyperboloidal_slicing_;
  p | order_;
}

bool operator==(const Sommerfeld& lhs, const Sommerfeld& rhs) {
  return lhs.black_hole_mass_ == rhs.black_hole_mass_ and
         lhs.black_hole_spin_ == rhs.black_hole_spin_ and
         lhs.orbital_radius_ == rhs.orbital_radius_ and
         lhs.m_mode_number_ == rhs.m_mode_number_ and
         lhs.hyperboloidal_slicing_ == rhs.hyperboloidal_slicing_ and
         lhs.order_ == rhs.order_;
}

bool operator!=(const Sommerfeld& lhs, const Sommerfeld& rhs) {
  return not(lhs == rhs);
}

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID Sommerfeld::my_PUP_ID = 0;  // NOLINT
#endif                                        // SPECTRE_USE_CHARM

}  // namespace ScalarSelfForce::BoundaryConditions
