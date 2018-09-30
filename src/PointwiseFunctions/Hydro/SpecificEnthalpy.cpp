// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

/// \cond
namespace hydro {
template <typename DataType>
Scalar<DataType> specific_enthalpy(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) noexcept {
  return Scalar<DataType>{1.0 + get(specific_internal_energy) +
                          get(pressure) / get(rest_mass_density)};
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                               \
  template Scalar<DTYPE(data)> specific_enthalpy(          \
      const Scalar<DTYPE(data)>& rest_mass_density,        \
      const Scalar<DTYPE(data)>& specific_internal_energy, \
      const Scalar<DTYPE(data)>& pressure) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace hydro
/// \endcond
