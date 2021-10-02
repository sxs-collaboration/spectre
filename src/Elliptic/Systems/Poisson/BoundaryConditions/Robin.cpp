// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/BoundaryConditions/Robin.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Options.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace Poisson::BoundaryConditions::detail {

RobinImpl::RobinImpl(const double dirichlet_weight, const double neumann_weight,
                     const double constant, const Options::Context& context)
    : dirichlet_weight_(dirichlet_weight),
      neumann_weight_(neumann_weight),
      constant_(constant) {
  if (dirichlet_weight == 0. and neumann_weight == 0.) {
    PARSE_ERROR(
        context,
        "Either the dirichlet_weight or the neumann_weight must be non-zero "
        "(or both).");
  }
}

double RobinImpl::dirichlet_weight() const { return dirichlet_weight_; }
double RobinImpl::neumann_weight() const { return neumann_weight_; }
double RobinImpl::constant() const { return constant_; }

void RobinImpl::apply(
    const gsl::not_null<Scalar<DataVector>*> field,
    const gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient) const {
  if (neumann_weight_ == 0.) {
    ASSERT(
        not equal_within_roundoff(dirichlet_weight_, 0.),
        "The dirichlet_weight is close to zero. Set it to a non-zero value to "
        "avoid divisions by small numbers.");
    get(*field) = constant_ / dirichlet_weight_;
  } else {
    ASSERT(not equal_within_roundoff(neumann_weight_, 0.),
           "The neumann_weight is close to zero. Set it to a non-zero value to "
           "avoid divisions by small numbers.");
    get(*n_dot_field_gradient) =
        (constant_ - dirichlet_weight_ * get(*field)) / neumann_weight_;
  }
}

void RobinImpl::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> field_correction,
    const gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient_correction)
    const {
  if (neumann_weight_ == 0.) {
    get(*field_correction) = 0.;
  } else {
    ASSERT(not equal_within_roundoff(neumann_weight_, 0.),
           "The neumann_weight is close to zero. Set it to a non-zero value to "
           "avoid divisions by small numbers.");
    get(*n_dot_field_gradient_correction) =
        -dirichlet_weight_ / neumann_weight_ * get(*field_correction);
  }
}

void RobinImpl::pup(PUP::er& p) {
  p | dirichlet_weight_;
  p | neumann_weight_;
  p | constant_;
}

bool operator==(const RobinImpl& lhs, const RobinImpl& rhs) {
  return lhs.dirichlet_weight() == rhs.dirichlet_weight() and
         lhs.neumann_weight() == rhs.neumann_weight() and
         lhs.constant() == rhs.constant();
}

bool operator!=(const RobinImpl& lhs, const RobinImpl& rhs) {
  return not(lhs == rhs);
}

}  // namespace Poisson::BoundaryConditions::detail
