// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/BoundaryCorrections/RegisterDerived.hpp"

#include "Evolution/Systems/ForceFree/BoundaryCorrections/Factory.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace ForceFree::BoundaryCorrections {
void register_derived_with_charm() {
  Parallel::register_derived_classes_with_charm<BoundaryCorrection>();
}
}  // namespace ForceFree::BoundaryCorrections
