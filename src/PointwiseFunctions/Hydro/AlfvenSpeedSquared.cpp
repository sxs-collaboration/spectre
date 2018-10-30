// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/AlfvenSpeedSquared.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

/// \cond
namespace hydro {
template <typename DataType>
Scalar<DataType> alfven_speed_squared(
    const Scalar<DataType>& comoving_magnetic_field_squared,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) noexcept {
  return Scalar<DataType>{get(comoving_magnetic_field_squared) /
                          (get(comoving_magnetic_field_squared) +
                           get(rest_mass_density) * get(specific_enthalpy))};
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template Scalar<DTYPE(data)> alfven_speed_squared(              \
      const Scalar<DTYPE(data)>& comoving_magnetic_field_squared, \
      const Scalar<DTYPE(data)>& rest_mass_density,               \
      const Scalar<DTYPE(data)>& specific_enthalpy) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace hydro
/// \endcond
