// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/Sommerfeld.hpp"

#include <blaze/Math.h>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace GrSelfForce::BoundaryConditions {

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

Sommerfeld::Sommerfeld(CkMigrateMessage* m) : Base(m) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Sommerfeld::get_clone() const {
  return std::make_unique<Sommerfeld>(*this);
}

void Sommerfeld::apply(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field,
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> n_dot_flux,
    const GradTensorType& /*deriv_field*/,
    const tnsr::I<ComplexDataVector, 2>& alpha,
    const tnsr::aaBB<ComplexDataVector, 3>& beta,
    const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar) const {
  if (hyperboloidal_slicing_) {
    if (order_ == 1) {
      for (size_t i = 0; i < field->size(); ++i) {
        (*n_dot_flux)[i] = 0.;
      }
    } else if (order_ == 2) {
      // 2nd-order Sommerfeld with hyperboloidal slicing imposes
      // d_x(alpha * d_x(Psi)) = 0, i.e. gamma . d_x(Psi) = -beta . Psi, where
      // x is r_star or r depending on PenetratingHorizon. We invert this
      // (dense in tensor-component space) for d_x(Psi), then set the flux
      // F^x = alpha^x d_x(Psi).
      using TensorStruct = std::decay_t<decltype(*field)>::structure;
      const size_t n_points = field->begin()->size();
      for (size_t i = 0; i < n_points; ++i) {
        blaze::StaticVector<std::complex<double>, 10> b_local(0.0);
        blaze::StaticMatrix<std::complex<double>, 10, 10> A_local(0.0);

        for (size_t a = 0; a < 4; ++a) {
          for (size_t b = 0; b <= a; ++b) {
            const size_t row = TensorStruct::get_storage_index(a, b);
            for (size_t c = 0; c < 4; ++c) {
              for (size_t d = 0; d <= c; ++d) {
                const size_t col = TensorStruct::get_storage_index(c, d);
                b_local[row] -= beta.get(a, b, c, d)[i] * field->get(c, d)[i];
                A_local(row, col) = gamma_rstar.get(a, b, c, d)[i];
              }
            }
          }
        }
        blaze::StaticVector<std::complex<double>, 10> grad_vec =
            blaze::solve(A_local, b_local);
        const std::complex<double> alpha_factor = get<0>(alpha)[i];
        for (size_t a = 0; a < 4; ++a) {
          for (size_t b = 0; b <= a; ++b) {
            const size_t row = TensorStruct::get_storage_index(a, b);
            n_dot_flux->get(a, b)[i] = alpha_factor * grad_vec[row];
          }
        }
      }

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
  if (order_ == 1) {
    for (size_t i = 0; i < field->size(); ++i) {
      (*n_dot_flux)[i] =
          std::complex<double>(0.0, m_mode_number_ * omega) * (*field)[i];
    }
  } else {
    ERROR("Order " << order_
                   << " not implemented for Sommerfeld boundary condition.");
  }
}

void Sommerfeld::apply_linearized(
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> field_correction,
    const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*>
        n_dot_flux_correction,
    const GradTensorType& deriv_field_correction,
    const tnsr::I<ComplexDataVector, 2>& alpha,
    const tnsr::aaBB<ComplexDataVector, 3>& beta,
    const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar) const {
  apply(field_correction, n_dot_flux_correction,
        deriv_field_correction, alpha, beta, gamma_rstar);
}

void Sommerfeld::pup(PUP::er& p) {
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

PUP::able::PUP_ID Sommerfeld::my_PUP_ID = 0;  // NOLINT

}  // namespace GrSelfForce::BoundaryConditions
