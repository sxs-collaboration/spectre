// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/IrrotationalBns/BoundaryConditions/Neumann.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
namespace IrrotationalBns::BoundaryConditions {

Neumann::Neumann(const Options::Context& /*context*/){};

void Neumann::apply(
    gsl::not_null<Scalar<DataVector>*> /*velocity_potential*/,
    gsl::not_null<Scalar<DataVector>*> n_dot_auxiliary_velocity) const {
  n_dot_auxiliary_velocity->get() = 0.0;
}

void Neumann::apply_linearized(
    gsl::not_null<Scalar<DataVector>*> /*velocity_potential_correction*/,
    gsl::not_null<Scalar<DataVector>*> n_dot_auxiliary_velocity_correction)
    const {
  n_dot_auxiliary_velocity_correction->get() = 0.0;
}

void Neumann::pup(PUP::er& /*p*/){};

bool operator==(const Neumann& /*lhs*/, const Neumann& /*rhs*/) { return true; }

bool operator!=(const Neumann& lhs, const Neumann& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID Neumann::my_PUP_ID = 0;  // NOLINT

}  // namespace IrrotationalBns::BoundaryConditions
