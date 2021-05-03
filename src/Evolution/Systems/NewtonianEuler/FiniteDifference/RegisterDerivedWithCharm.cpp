// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace NewtonianEuler::fd {
void register_derived_with_charm() noexcept {
  Parallel::register_classes_with_charm(
      typename Reconstructor<1>::creatable_classes{});
  Parallel::register_classes_with_charm(
      typename Reconstructor<2>::creatable_classes{});
  Parallel::register_classes_with_charm(
      typename Reconstructor<3>::creatable_classes{});
}
}  // namespace NewtonianEuler::fd
