// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/ShuOsherTube.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct ShuOsherTubeProxy : NewtonianEuler::AnalyticData::ShuOsherTube {
  using NewtonianEuler::AnalyticData::ShuOsherTube::ShuOsherTube;

  template <typename DataType>
  using variables_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensity<DataType>,
                 NewtonianEuler::Tags::Velocity<DataType, 1, Frame::Inertial>,
                 NewtonianEuler::Tags::SpecificInternalEnergy<DataType>,
                 NewtonianEuler::Tags::Pressure<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, 1, Frame::Inertial>& x) const {
    return this->variables(x, variables_tags<DataType>{});
  }
};

template <size_t Dim, typename DataType>
void test_analytic_data(const DataType& used_for_size) {
  const double mass_density_l = 3.857143;
  const double velocity_l = 2.629369;
  const double pressure_l = 10.33333;
  const double jump_position = -4.0;
  const double epsilon = 0.2;
  const double lambda = 5.0;
  const double velocity_r = 0.0;
  const double pressure_r = 1.0;
  const double adiabatic_index = 1.4;
  const auto members =
      std::make_tuple(mass_density_l, velocity_l, pressure_l, jump_position,
                      epsilon, lambda, velocity_r, pressure_r, adiabatic_index);

  ShuOsherTubeProxy shu_osher(jump_position, mass_density_l, velocity_l,
                              pressure_l, velocity_r, pressure_r, epsilon,
                              lambda);
  pypp::check_with_random_values<1>(
      &ShuOsherTubeProxy::template primitive_variables<DataType>, shu_osher,
      "ShuOsherTube",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      {{{0.0, 1.0}}}, members, used_for_size);

  const auto shu_osher_from_options =
      TestHelpers::test_creation<NewtonianEuler::AnalyticData::ShuOsherTube>(
          "  JumpPosition: -4.0\n"
          "  LeftMassDensity: 3.857143\n"
          "  LeftVelocity: 2.629369\n"
          "  LeftPressure: 10.33333\n"
          "  RightVelocity: 0.0\n"
          "  RightPressure: 1.0\n"
          "  Epsilon: 0.2\n"
          "  Lambda: 5.0\n");
  CHECK(shu_osher_from_options == shu_osher);

  // run post-serialized state through checks with random numbers
  pypp::check_with_random_values<1>(
      &ShuOsherTubeProxy::template primitive_variables<DataType>,
      serialize_and_deserialize(shu_osher), "ShuOsherTube",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      {{{0.0, 1.0}}}, members, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.NewtEuler.ShuOsherTube",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_analytic_data, (1));
}
