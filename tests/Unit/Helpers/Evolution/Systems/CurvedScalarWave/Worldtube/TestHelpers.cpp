// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Evolution/Systems/CurvedScalarWave/Worldtube/TestHelpers.hpp"

#include <cstddef>
#include <sstream>
#include <string>

#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"

namespace TestHelpers::CurvedScalarWave::Worldtube {
std::unique_ptr<DomainCreator<3>> worldtube_binary_compact_object(
    const double orbit_radius, const double worldtube_radius,
    const double angular_velocity) {
  ASSERT(orbit_radius > 4. * worldtube_radius,
         "Can't construct a domain with these parameters");
  std::stringstream angular_velocity_stream;
  angular_velocity_stream << std::fixed << std::setprecision(16)
                          << angular_velocity;
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
      "    RadialDistribution: Linear\n"
      "  OuterShell:\n"
      "    Radius: 50.0\n"
      "    RadialDistribution: Linear\n"
      "    OpeningAngle: 90.0\n"
      "    BoundaryCondition:\n"
      "      None\n"
      "  InitialRefinement: 0\n"
      "  InitialGridPoints: 3\n"
      "  UseEquiangularMap: true\n"
      "  TimeDependentMaps:\n"
      "    InitialTime: 0.0\n"
      "    ExpansionMap: \n"
      "      InitialValues: [1.0, 0.0]\n"
      "      AsymptoticVelocityOuterBoundary: 0.0\n"
      "      DecayTimescaleOuterBoundaryVelocity: 1.0\n"
      "    RotationMap:\n"
      "      InitialAngularVelocity: [0.0, 0.0," +
      angular_velocity_stream.str() +
      "]\n"
      "    SizeMapA:\n"
      "      InitialValues: [0.0, 0.0, 0.0]\n"
      "    SizeMapB:\n"
      "      InitialValues: [0.0, 0.0, 0.0]\n"
      "    ShapeMapA:\n"
      "      LMax: 8\n"
      "    ShapeMapB:\n"
      "      LMax: 8";
  return ::TestHelpers::test_option_tag<::domain::OptionTags::DomainCreator<3>,
                                        Metavariables>(
      binary_compact_object_options);
}
}  // namespace TestHelpers::CurvedScalarWave::Worldtube
