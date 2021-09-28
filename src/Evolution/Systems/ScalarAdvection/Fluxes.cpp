// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarAdvection {
template <size_t Dim>
void Fluxes<Dim>::apply(const gsl::not_null<tnsr::I<DataVector, Dim>*> u_flux,
                        const Scalar<DataVector>& u,
                        const tnsr::I<DataVector, Dim>& velocity_field) {
  for (size_t i = 0; i < Dim; ++i) {
    u_flux->get(i) = velocity_field.get(i) * get(u);
  }
}

}  // namespace ScalarAdvection

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class ScalarAdvection::Fluxes<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2))

#undef DIM
#undef INSTANTIATE
