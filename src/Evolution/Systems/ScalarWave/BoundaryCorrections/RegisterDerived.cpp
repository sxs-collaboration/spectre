// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/BoundaryCorrections/RegisterDerived.hpp"

#include "Evolution/Systems/ScalarWave/BoundaryCorrections/Factory.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace ScalarWave::BoundaryCorrections {
void register_derived_with_charm() {
  register_derived_classes_with_charm<BoundaryCorrection<1>>();
  register_derived_classes_with_charm<BoundaryCorrection<2>>();
  register_derived_classes_with_charm<BoundaryCorrection<3>>();
}
}  // namespace ScalarWave::BoundaryCorrections
