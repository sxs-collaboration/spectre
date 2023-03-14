// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Protocols/Metavariables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::CurvedScalarWave::Worldtube {
struct Metavariables {
  struct domain : tt::ConformsTo<::domain::protocols::Metavariables> {
    static constexpr bool enable_time_dependent_maps = true;
  };
  using system =
      TestHelpers::domain::BoundaryConditions::SystemWithBoundaryConditions<3>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        DomainCreator<3>, tmpl::list<::domain::creators::BinaryCompactObject>>>;
  };
};

// returns a binary compact object domain creator with the excision spheres
// corresponding to the circular worldtube setup: a central excision sphere
// (the central black hole) and a worldtube excision sphere in circular orbit
// around it with angular velocity R^{-3/2}, where R is the orbital radius.
std::unique_ptr<DomainCreator<3>> worldtube_binary_compact_object(
    const double orbit_radius, const double worldtube_radius,
    const double angular_velocity);

}  // namespace TestHelpers::CurvedScalarWave::Worldtube
