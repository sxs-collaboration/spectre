// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/RegisterDerived.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
void register_derived_with_charm() {
  register_derived_classes_with_charm<BoundaryCorrection>();
}
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
