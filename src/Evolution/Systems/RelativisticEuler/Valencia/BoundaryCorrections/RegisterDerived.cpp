// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryCorrections/RegisterDerived.hpp"

#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryCorrections/Factory.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace RelativisticEuler::Valencia::BoundaryCorrections {
void register_derived_with_charm() {
  register_derived_classes_with_charm<BoundaryCorrection<1>>();
  register_derived_classes_with_charm<BoundaryCorrection<2>>();
  register_derived_classes_with_charm<BoundaryCorrection<3>>();
}
}  // namespace RelativisticEuler::Valencia::BoundaryCorrections
