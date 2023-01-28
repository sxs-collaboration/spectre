// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Punctures/AdmIntegrals.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Punctures {

void adm_mass_integrand(const gsl::not_null<Scalar<DataVector>*> result,
                        const Scalar<DataVector>& field,
                        const Scalar<DataVector>& alpha,
                        const Scalar<DataVector>& beta) {
  get(*result) = get(alpha) * (get(field) + 1.) + 1.;
  get(*result) = pow<7>(get(*result));
  get(*result) = 0.5 / M_PI * get(beta) / get(*result);
}

Scalar<DataVector> adm_mass_integrand(const Scalar<DataVector>& field,
                                      const Scalar<DataVector>& alpha,
                                      const Scalar<DataVector>& beta) {
  Scalar<DataVector> result{get(field).size()};
  adm_mass_integrand(make_not_null(&result), field, alpha, beta);
  return result;
}

}  // namespace Punctures
