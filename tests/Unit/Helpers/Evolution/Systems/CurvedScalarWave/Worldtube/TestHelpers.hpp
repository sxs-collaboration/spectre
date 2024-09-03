// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::CurvedScalarWave::Worldtube {
template <bool EvolveOrbit>
struct Metavariables {
  using system =
      TestHelpers::domain::BoundaryConditions::SystemWithBoundaryConditions<3>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    // we set `UseWorldtube` to `false` here so the functions of time are valid
    // which simplifies testing.
    using factory_classes = tmpl::map<tmpl::pair<
        DomainCreator<3>,
        tmpl::list<::domain::creators::BinaryCompactObject<EvolveOrbit>>>>;
  };
};

// returns a binary compact object domain creator with the excision spheres
// corresponding to the circular worldtube setup: a central excision sphere
// (the central black hole) and a worldtube excision sphere in circular orbit
// around it with angular velocity R^{-3/2}, where R is the orbital radius.
template <bool EvolveOrbit>
std::unique_ptr<DomainCreator<3>> worldtube_binary_compact_object(
    const double orbit_radius, const double worldtube_radius,
    const double angular_velocity);

}  // namespace TestHelpers::CurvedScalarWave::Worldtube
