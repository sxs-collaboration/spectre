// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryCorrections {
void register_derived_with_charm() noexcept {
  Parallel::register_derived_classes_with_charm<BoundaryCorrection>();
}
}  // namespace grmhd::GhValenciaDivClean::BoundaryCorrections
