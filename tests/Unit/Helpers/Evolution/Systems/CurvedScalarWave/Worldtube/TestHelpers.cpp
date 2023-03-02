// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Evolution/Systems/CurvedScalarWave/Worldtube/TestHelpers.hpp"

#include <cstddef>
#include <string>

#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"

namespace TestHelpers::CurvedScalarWave::Worldtube {
std::unique_ptr<DomainCreator<3>> worldtube_binary_compact_object(
    const double orbit_radius, const double worldtube_radius) {
  ASSERT(orbit_radius > 4. * worldtube_radius,
         "Can't construct a domain with these parameters");
  const double angular_velocity = 1. / (orbit_radius * sqrt(orbit_radius));
  std::string binary_compact_object_options =
      "BinaryCompactObject:\n"
      "  ObjectA:\n"
      "    InnerRadius: " +
      std::to_string(worldtube_radius) +
      "\n"
      "    OuterRadius: " +
      std::to_string(2. * worldtube_radius) +
      "\n"
      "    XCoord: " +
      std::to_string(orbit_radius) +
      "\n"
      "    Interior:\n"
      "      ExciseWithBoundaryCondition:\n"
      "        None:\n"
      "    UseLogarithmicMap: false\n"
      "  ObjectB:\n"
      "    InnerRadius: 1.9\n"
      "    OuterRadius: 3.0\n"
      "    XCoord: -1e-99\n"
      "    Interior:\n"
      "      ExciseWithBoundaryCondition:\n"
      "        None\n"
      "    UseLogarithmicMap: false\n"
      "  Envelope:\n"
      "    Radius: 30.0\n"
      "    UseProjectiveMap: false\n"
      "  OuterShell:\n"
      "    Radius: 50.0\n"
      "    RadialDistribution: Linear\n"
      "    BoundaryCondition:\n"
      "      None\n"
      "  InitialRefinement: 0\n"
      "  InitialGridPoints: 3\n"
      "  UseEquiangularMap: true\n"
      "  TimeDependentMaps:\n"
      "    InitialTime: 0.0\n"
      "    ExpansionMap: \n"
      "      OuterBoundary: 1.0\n"
      "      InitialExpansion: 1.0\n"
      "      InitialExpansionVelocity: 0.0\n"
      "      AsymptoticVelocityOuterBoundary: 0.0\n"
      "      DecayTimescaleOuterBoundaryVelocity: 1.0\n"
      "    RotationMap:\n"
      "      InitialAngularVelocity: [0.0, 0.0," +
      std::to_string(angular_velocity) +
      "]\n"
      "    SizeMap:\n"
      "      InitialValues: [0.0, 0.0]\n"
      "      InitialVelocities: [0.0, 0.0]\n"
      "      InitialAccelerations: [0.0, 0.0]";
  return ::TestHelpers::test_option_tag<::domain::OptionTags::DomainCreator<3>,
                                        Metavariables>(
      binary_compact_object_options);
}
}  // namespace TestHelpers::CurvedScalarWave::Worldtube
