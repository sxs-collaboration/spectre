// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"

namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.RadiationTransport.M1Grey.BoundaryConditions.Periodic",
                  "[Unit][Evolution]") {
  using boundary_condition =
      RadiationTransport::M1Grey::BoundaryConditions::BoundaryCondition<
          neutrinos::ElectronNeutrinos<1>, neutrinos::ElectronAntiNeutrinos<1>>;
  helpers::test_periodic_condition<
      domain::BoundaryConditions::Periodic<boundary_condition>,
      boundary_condition>("Periodic:\n");
}
