// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <optional>
#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::AnalyticData {
namespace {

using test_tags = tmpl::list<
    Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
    Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
    Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                            Frame::Inertial>,
    Tags::ConformalFactorMinusOne<DataVector>,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataVector>, 0>,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataVector, 3>, 0>>;

template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
struct BinaryWithGravitationalWavesProxy {
  tuples::tagged_tuple_from_typelist<test_tags> test_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
    return binary_with_gravitational_waves->variables(x, test_tags{});
  }

  BinaryWithGravitationalWaves<IsolatedObjectBase, IsolatedObjectClasses>*
      binary_with_gravitational_waves;
};

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<elliptic::analytic_data::Background,
                             tmpl::list<BinaryWithGravitationalWaves<
                                 elliptic::analytic_data::AnalyticSolution,
                                 tmpl::list<Xcts::Solutions::Schwarzschild>>>>,
                  tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                             tmpl::list<Xcts::Solutions::Schwarzschild>>>;
  };
};

void test_data(const std::array<double, 2>& x_coords,
               const std::array<double, 2>& masses,
               const std::array<double, 3>& momentum_left,
               const std::array<double, 3>& momentum_right,
               const std::array<double, 2>& center_of_mass_offset,
               const double& attenuation_parameter,
               const double& attenuation_radius,
               const std::string& py_functions_suffix,
               const std::string& options_string) {
  using IsolatedObjectBase = elliptic::analytic_data::AnalyticSolution;
  using IsolatedObjectClasses = tmpl::list<Xcts::Solutions::Schwarzschild>;
  register_classes_with_charm<Xcts::Solutions::Schwarzschild>();
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<elliptic::analytic_data::Background>, Metavariables>(
      options_string);
  REQUIRE(dynamic_cast<const BinaryWithGravitationalWaves<
              IsolatedObjectBase, IsolatedObjectClasses>*>(created.get()) !=
          nullptr);
  const auto& derived = dynamic_cast<const BinaryWithGravitationalWaves<
      IsolatedObjectBase, IsolatedObjectClasses>&>(*created);
  auto BinaryWithGravitationalWaves = serialize_and_deserialize(derived);
  {
    INFO("Properties");
    CHECK(BinaryWithGravitationalWaves.x_coords() == x_coords);
    CHECK(BinaryWithGravitationalWaves.y_offset() == center_of_mass_offset[0]);
    CHECK(BinaryWithGravitationalWaves.z_offset() == center_of_mass_offset[1]);
    CHECK(BinaryWithGravitationalWaves.momentum_left() == momentum_left);
    CHECK(BinaryWithGravitationalWaves.momentum_right() == momentum_right);
    const auto& superposed_objects =
        BinaryWithGravitationalWaves.superposed_objects();
    CHECK(dynamic_cast<const Xcts::Solutions::Schwarzschild&>(
              *superposed_objects[0])
              .mass() == masses[0]);
    CHECK(dynamic_cast<const Xcts::Solutions::Schwarzschild&>(
              *superposed_objects[1])
              .mass() == masses[1]);
    CHECK(BinaryWithGravitationalWaves.attenuation_parameter() ==
          attenuation_parameter);
    CHECK(BinaryWithGravitationalWaves.attenuation_radius() ==
          attenuation_radius);
  }
  {
    const BinaryWithGravitationalWavesProxy<IsolatedObjectBase,
                                            IsolatedObjectClasses>
        proxy{&BinaryWithGravitationalWaves};
    pypp::check_with_random_values<1>(
        &BinaryWithGravitationalWavesProxy<
            IsolatedObjectBase, IsolatedObjectClasses>::test_variables,
        proxy, "BinaryWithGravitationalWaves",
        {"conformal_metric_" + py_functions_suffix,
         "inv_conformal_metric_" + py_functions_suffix, "shift_background",
         "longitudinal_shift_background_" + py_functions_suffix,
         "conformal_factor_minus_one_" + py_functions_suffix,
         "energy_density_" + py_functions_suffix,
         "stress_trace_" + py_functions_suffix,
         "momentum_density_" + py_functions_suffix},
        {{{x_coords[0] * 2, x_coords[1] * 2}}}, std::make_tuple(),
        DataVector(5));
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.Xcts.BinaryWithGravitationalWaves",
    "[PointwiseFunctions][Unit]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/Xcts"};
  test_data({{-5., 6.}}, {{1.1, 0.43}}, {{-0.01, -0.01, -0.01}},
            {{0.01, 0.01, 0.01}}, {{0.02, 0.01}}, 1.1, 3.6, "bbh_isotropic",
            "BinaryWithGravitationalWaves:\n"
            "  XCoords: [-5., 6.]\n"
            "  Masses: [1.1, 0.43]\n"
            "  MomentumLeft: [-0.01, -0.01, -0.01]\n"
            "  MomentumRight: [0.01, 0.01, 0.01]\n"
            "  CenterOfMassOffset: [0.02, 0.01]\n"
            "  ObjectLeft:\n"
            "    Schwarzschild:\n"
            "      Mass: 1.1\n"
            "      Coordinates: Isotropic\n"
            "  ObjectRight:\n"
            "    Schwarzschild:\n"
            "      Mass: 0.43\n"
            "      Coordinates: Isotropic\n"
            "  AttenuationParameter: 1.1\n"
            "  AttenuationRadius: 3.6");
}

}  // namespace Xcts::AnalyticData
