// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/Sources/ScalarSource.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace ScalarTensor {

void add_scalar_source_to_dt_pi_scalar(
    gsl::not_null<Scalar<DataVector>*> dt_pi_scalar,
    const Scalar<DataVector>& scalar_source, const Scalar<DataVector>& lapse) {
  get(*dt_pi_scalar) += get(lapse) * get(scalar_source);
}

void mass_source(
    const gsl::not_null<Scalar<DataVector>*> scalar_source,
    const Scalar<DataVector>& psi, const double mass_psi) {
  get(*scalar_source) = square(mass_psi) * get(psi);
}
}  // namespace ScalarTensor
