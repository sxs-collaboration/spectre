// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace Burgers::BoundaryCorrections {
void register_derived_with_charm() {
  Parallel::register_derived_classes_with_charm<BoundaryCorrection>();
}
}  // namespace Burgers::BoundaryCorrections
