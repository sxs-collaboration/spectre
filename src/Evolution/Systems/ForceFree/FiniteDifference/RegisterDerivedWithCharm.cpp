// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/FiniteDifference/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/ForceFree/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace ForceFree::fd {
void register_derived_with_charm() {
  register_classes_with_charm(typename Reconstructor::creatable_classes{});
}
}  // namespace ForceFree::fd
