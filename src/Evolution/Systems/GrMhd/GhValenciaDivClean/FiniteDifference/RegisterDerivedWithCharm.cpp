// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace grmhd::GhValenciaDivClean::fd {
void register_derived_with_charm() {
  Parallel::register_classes_with_charm(
      typename Reconstructor::creatable_classes{});
}
}  // namespace grmhd::GhValenciaDivClean::fd
