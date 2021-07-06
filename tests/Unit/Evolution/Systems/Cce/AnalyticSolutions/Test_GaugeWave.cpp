// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <memory>

#include "Evolution/Systems/Cce/AnalyticSolutions/GaugeWave.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/AnalyticSolutions/AnalyticDataHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace Cce::Solutions {
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.GaugeWave", "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> radius_dist{5.0, 10.0};
  UniformCustomDistribution<double> parameter_dist{0.5, 2.0};
  const double extraction_radius = radius_dist(gen);
  const size_t l_max = 16;
  // use a low frequency so that the dimensionless parameter in the metric is ~1
  const double frequency = parameter_dist(gen) / 5.0;
  const double mass = parameter_dist(gen);
  const double amplitude = parameter_dist(gen) / 10.0;
  const double peak_time = parameter_dist(gen);
  const double duration = parameter_dist(gen);
  TestHelpers::SphericalSolutionWrapper<GaugeWave> boundary_solution{
      extraction_radius, mass, frequency, amplitude, peak_time, duration};
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Cce/AnalyticSolutions/"};
  boundary_solution.test_spherical_metric(
      "GaugeWave", l_max, peak_time - 0.1 * duration,
      Approx::custom().epsilon(1.e-11).scale(1.0), extraction_radius, mass,
      frequency, amplitude, peak_time, duration);
  boundary_solution.test_serialize_and_deserialize(l_max,
                                                   peak_time - 0.1 * duration);
  TestHelpers::test_initialize_j(
      l_max, 5_st, extraction_radius, peak_time - 0.1 * duration,
      std::make_unique<InitializeJ::InverseCubic<false>>(),
      boundary_solution.get_clone());
}
}  // namespace Cce::Solutions
