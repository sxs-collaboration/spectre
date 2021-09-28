// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace grmhd::ValenciaDivClean::fd {
void register_derived_with_charm() {
  Parallel::register_classes_with_charm(
      typename Reconstructor::creatable_classes{});
}
}  // namespace grmhd::ValenciaDivClean::fd
