// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/Factory.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace CurvedScalarWave::BoundaryCorrections {
void register_derived_with_charm() noexcept {
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<1>>();
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<2>>();
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<3>>();
}
}  // namespace CurvedScalarWave::BoundaryCorrections
