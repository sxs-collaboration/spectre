// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/FiniteDifference/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace ScalarAdvection::fd {
void register_derived_with_charm() {
  register_derived_classes_with_charm<Reconstructor<1>>();
  register_derived_classes_with_charm<Reconstructor<2>>();
}
}  // namespace ScalarAdvection::fd
