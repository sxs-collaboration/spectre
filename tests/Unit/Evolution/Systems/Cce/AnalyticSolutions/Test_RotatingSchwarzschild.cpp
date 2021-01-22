// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>

#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/TeukolskyWave.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/AnalyticSolutions/AnalyticDataHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace Cce::Solutions {
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.RotatingSchwarzschild",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> radius_dist{5.0, 10.0};
  UniformCustomDistribution<double> parameter_dist{0.5, 2.0};
  const double extraction_radius = radius_dist(gen);
  const size_t l_max = 16;
  // use a low frequency so that the dimensionless parameter in the metric is ~1
  const double frequency = parameter_dist(gen) / 30.0;
  const double mass = parameter_dist(gen);
  TestHelpers::SphericalSolutionWrapper<RotatingSchwarzschild>
      boundary_solution{extraction_radius, mass, frequency};
  const double time = 10.0 * parameter_dist(gen);
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Cce/AnalyticSolutions/"};
  boundary_solution.test_spherical_metric("RotatingSchwarzschild", l_max, time,
                                          approx, extraction_radius, mass,
                                          frequency);
  boundary_solution.test_serialize_and_deserialize(l_max, time);
}
}  // namespace Cce::Solutions
