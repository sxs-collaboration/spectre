// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/InversePlasmaBeta.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {
template <typename DataType>
void inverse_plasma_beta(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const Scalar<DataType>& fluid_pressure) {
  get(*result) =
      0.5*square(get(comoving_magnetic_field_magnitude))/get(fluid_pressure);
}

template <typename DataType>
Scalar<DataType> inverse_plasma_beta(
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const Scalar<DataType>& fluid_pressure) {
  Scalar<DataType> result{};
  inverse_plasma_beta(make_not_null(&result), comoving_magnetic_field_magnitude,
                      fluid_pressure);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template void inverse_plasma_beta(                              \
      gsl::not_null<Scalar<DTYPE(data)>*> result,                 \
      const Scalar<DTYPE(data)>& comoving_magnetic_field_magnitude, \
      const Scalar<DTYPE(data)>& fluid_pressure);                 \
  template Scalar<DTYPE(data)> inverse_plasma_beta(               \
      const Scalar<DTYPE(data)>& comoving_magnetic_field_magnitude, \
      const Scalar<DTYPE(data)>& fluid_pressure);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
}  // namespace hydro
