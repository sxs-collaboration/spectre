// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/RegisterDerived.hpp"

#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/Factory.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace ScalarAdvection::BoundaryCorrections {
void register_derived_with_charm() {
  register_derived_classes_with_charm<BoundaryCorrection<1>>();
  register_derived_classes_with_charm<BoundaryCorrection<2>>();
}
}  // namespace ScalarAdvection::BoundaryCorrections
