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
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::AnalyticData {
namespace {

using test_tags = tmpl::list<
    Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
    Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
    ::Tags::deriv<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                            Frame::Inertial>,
    Tags::ConformalFactor<DataVector>>;

template <typename IsolatedObjectRegistrars>
struct BinaryProxy {
  tuples::tagged_tuple_from_typelist<test_tags> test_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return binary.variables(x, test_tags{});
  }

  const Binary<IsolatedObjectRegistrars>& binary;
};

template <typename IsolatedObjectRegistrars>
void test_data(const std::array<double, 2>& x_coords,
               const double angular_velocity,
               const std::optional<std::array<double, 2>>& falloff_widths,
               const std::array<double, 2>& masses,
               const std::string& py_functions_suffix,
               const std::string& options_string) {
  const auto created = TestHelpers::test_factory_creation<
      ::AnalyticData<3, tmpl::list<Registrars::Binary<tmpl::list<
                            Xcts::Solutions::Registrars::Schwarzschild>>>>>(
      options_string);
  REQUIRE(dynamic_cast<const Binary<
              tmpl::list<Xcts::Solutions::Registrars::Schwarzschild>>*>(
              created.get()) != nullptr);
  const auto& derived = dynamic_cast<
      const Binary<tmpl::list<Xcts::Solutions::Registrars::Schwarzschild>>&>(
      *created);
  const auto binary = serialize_and_deserialize(derived);
  {
    INFO("Properties");
    CHECK(binary.x_coords() == x_coords);
    CHECK(binary.angular_velocity() == angular_velocity);
    CHECK(binary.falloff_widths() == falloff_widths);
    const auto& superposed_objects = binary.superposed_objects();
    CHECK(dynamic_cast<const Xcts::Solutions::Schwarzschild<>&>(
              *superposed_objects[0])
              .mass() == masses[0]);
    CHECK(dynamic_cast<const Xcts::Solutions::Schwarzschild<>&>(
              *superposed_objects[1])
              .mass() == masses[1]);
  }
  {
    const BinaryProxy<IsolatedObjectRegistrars> proxy{binary};
    pypp::check_with_random_values<1>(
        &BinaryProxy<IsolatedObjectRegistrars>::test_variables, proxy, "Binary",
        {"conformal_metric_" + py_functions_suffix,
         "inv_conformal_metric_" + py_functions_suffix,
         "deriv_conformal_metric_" + py_functions_suffix,
         "extrinsic_curvature_trace_" + py_functions_suffix, "shift_background",
         "longitudinal_shift_background_" + py_functions_suffix,
         "conformal_factor_" + py_functions_suffix},
        {{{x_coords[0] * 2, x_coords[1] * 2}}}, std::make_tuple(),
        DataVector(5));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.Xcts.Binary",
                  "[PointwiseFunctions][Unit]") {
  Parallel::register_classes_with_charm<Xcts::Solutions::Schwarzschild<>>();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/Xcts"};
  test_data<tmpl::list<Xcts::Solutions::Registrars::Schwarzschild>>(
      {{-5., 6.}}, 0.02, {{7., 8.}}, {{1.1, 0.43}}, "bbh_isotropic",
      "Binary:\n"
      "  XCoords: [-5., 6.]\n"
      "  ObjectA:\n"
      "    Schwarzschild:\n"
      "      Mass: 1.1\n"
      "      Coordinates: Isotropic\n"
      "  ObjectB:\n"
      "    Schwarzschild:\n"
      "      Mass: 0.43\n"
      "      Coordinates: Isotropic\n"
      "  AngularVelocity: 0.02\n"
      "  FalloffWidths: [7., 8.]");
}

}  // namespace Xcts::AnalyticData
