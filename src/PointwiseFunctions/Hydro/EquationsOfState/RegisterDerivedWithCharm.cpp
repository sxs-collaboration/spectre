// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"

#include <cstddef>

#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
namespace EquationsOfState {

template <bool IsRelativistic, size_t ThermodynamicDim>
void register_derived_subset_with_charm() {
  using derived_equations_of_state =
      typename EquationsOfState::detail::DerivedClasses<IsRelativistic,
                                                        ThermodynamicDim>::type;
  register_classes_with_charm(derived_equations_of_state{});
}

void register_derived_with_charm() {
  register_derived_subset_with_charm<true, 1>();
  register_derived_subset_with_charm<false, 1>();
  register_derived_subset_with_charm<true, 2>();
  register_derived_subset_with_charm<false, 2>();
}
}  // namespace EquationsOfState
