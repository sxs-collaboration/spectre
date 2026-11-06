// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/FiniteDifference/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace NewtonianMhd::fd {
void register_derived_with_charm() {
  register_classes_with_charm(typename Reconstructor<1>::creatable_classes{});
  register_classes_with_charm(typename Reconstructor<2>::creatable_classes{});
  register_classes_with_charm(typename Reconstructor<3>::creatable_classes{});
}
}  // namespace NewtonianMhd::fd
