// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/RegisterDerived.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryCorrections {
void register_derived_with_charm() {
  register_derived_classes_with_charm<BoundaryCorrection>();
}
}  // namespace grmhd::GhValenciaDivClean::BoundaryCorrections
