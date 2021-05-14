// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/TimeDerivativeTerms.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarAdvection {

template <size_t Dim>
void TimeDerivativeTerms<Dim>::apply(
    gsl::not_null<Scalar<DataVector>*> /*non_flux_terms_dt_vars*/,
    gsl::not_null<tnsr::I<DataVector, Dim>*> flux, const Scalar<DataVector>& u,
    const tnsr::I<DataVector, Dim>& velocity_field) noexcept {
  Fluxes<Dim>::apply(flux, u, velocity_field);
}

}  // namespace ScalarAdvection

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template class ScalarAdvection::TimeDerivativeTerms<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2))

#undef DIM
#undef INSTANTIATE
