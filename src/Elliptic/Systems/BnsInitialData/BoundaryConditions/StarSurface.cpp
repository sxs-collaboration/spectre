// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/BnsInitialData/BoundaryConditions/StarSurface.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
namespace BnsInitialData::BoundaryConditions {

StarSurface::StarSurface(const Options::Context& /*context*/) {}

void StarSurface::apply(
    gsl::not_null<Scalar<DataVector>*> /*velocity_potential*/,
    gsl::not_null<Scalar<DataVector>*> n_dot_flux_for_potential,
    const tnsr::i<DataVector, 3>& /*deriv_velocity_potential*/,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3>& rotational_shift,
    const double euler_enthalpy_constant,
    const tnsr::i<DataVector, 3>& normal) {
  tenex::evaluate<>(n_dot_flux_for_potential,
                    euler_enthalpy_constant / square(lapse()) *
                        rotational_shift(ti::I) * normal(ti::i));
}
// The term on the RHS is constant w.r.t. the variables, so the linearized form
// of the RHS is zero.
void StarSurface::apply_linearized(
    gsl::not_null<Scalar<DataVector>*> velocity_potential_correction,
    gsl::not_null<Scalar<DataVector>*> n_dot_flux_for_potential_correction,
    const tnsr::i<DataVector, 3>& /*deriv_velocity_potential*/) {
  set_number_of_grid_points(n_dot_flux_for_potential_correction,
                            *velocity_potential_correction);
  get(*n_dot_flux_for_potential_correction) = 0.0;
}

void StarSurface::pup(PUP::er& /*p*/) {}

bool operator==(const StarSurface& /*lhs*/, const StarSurface& /*rhs*/) {
  return true;
}

bool operator!=(const StarSurface& lhs, const StarSurface& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID StarSurface::my_PUP_ID = 0;  // NOLINT

}  // namespace BnsInitialData::BoundaryConditions
