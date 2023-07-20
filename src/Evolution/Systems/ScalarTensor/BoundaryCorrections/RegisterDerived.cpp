// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/RegisterDerived.hpp"

#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/Factory.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace ScalarTensor::BoundaryCorrections {
void register_derived_with_charm() {
  register_derived_classes_with_charm<BoundaryCorrection>();
}
}  // namespace ScalarTensor::BoundaryCorrections
