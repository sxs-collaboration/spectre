// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Characteristics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarAdvection::Tags {

template <size_t Dim>
void LargestCharacteristicSpeedCompute<Dim>::function(
    const gsl::not_null<double*> speed,
    const tnsr::I<DataVector, Dim>& velocity_field) noexcept {
  *speed = max(get(magnitude<DataVector>(velocity_field)));
}

}  // namespace ScalarAdvection::Tags

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void ScalarAdvection::Tags::LargestCharacteristicSpeedCompute<DIM( \
      data)>::function(gsl::not_null<double*> speed,                          \
                       const tnsr::I<DataVector, DIM(data)>&                  \
                           velocity_field) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2))

#undef DIM
#undef INSTANTIATE
