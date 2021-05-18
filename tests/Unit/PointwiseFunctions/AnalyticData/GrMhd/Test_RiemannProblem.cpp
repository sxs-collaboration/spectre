// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/RiemannProblem.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct RiemannProblemProxy : grmhd::AnalyticData::RiemannProblem {
  using grmhd::AnalyticData::RiemannProblem::RiemannProblem;

  template <typename DataType>
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  using grmhd_variables_tags =
      tmpl::push_back<hydro_variables_tags<DataType>,
                      hydro::Tags::MagneticField<DataType, 3>,
                      hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<hydro_variables_tags<DataType>>
  hydro_variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x) const noexcept {
    return variables(x, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x) const noexcept {
    return variables(x, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() {
  const auto riemann_problem =
      TestHelpers::test_creation<grmhd::AnalyticData::RiemannProblem>(
          "AdiabaticIndex: 2.0\n"
          "LeftDensity: 1.0\n"
          "LeftPressure: 1.0\n"
          "LeftVelocity: [0.0, 0.0, 0.0]\n"
          "LeftMagneticField: [0.5, 1.0, 0.0]\n"
          "RightDensity: 0.125\n"
          "RightPressure: 0.1\n"
          "RightVelocity: [0.0, 0.0, 0.0]\n"
          "RightMagneticField: [0.5, -1.0, 0.0]\n"
          "Lapse: 2.0\n"
          "ShiftX: 0.4\n");
  CHECK(riemann_problem ==
        grmhd::AnalyticData::RiemannProblem(
            2.0, 1.0, 0.125, 1.0, 0.1, std::array{0.0, 0.0, 0.0},
            std::array{0.0, 0.0, 0.0}, std::array{0.5, 1.0, 0.0},
            std::array{0.5, -1.0, 0.0}, 2.0, 0.4));
}

void test_move() {
  grmhd::AnalyticData::RiemannProblem riemann_problem(
      2.0, 1.0, 0.125, 1.0, 0.1, std::array{0.0, 0.0, 0.0},
      std::array{0.0, 0.0, 0.0}, std::array{0.5, 1.0, 0.0},
      std::array{0.5, -1.0, 0.0}, 2.0, 0.4);
  grmhd::AnalyticData::RiemannProblem riemann_problem_copy(
      2.0, 1.0, 0.125, 1.0, 0.1, std::array{0.0, 0.0, 0.0},
      std::array{0.0, 0.0, 0.0}, std::array{0.5, 1.0, 0.0},
      std::array{0.5, -1.0, 0.0}, 2.0, 0.4);
  test_move_semantics(std::move(riemann_problem),
                      riemann_problem_copy);  //  NOLINT
}

void test_serialize() {
  grmhd::AnalyticData::RiemannProblem riemann_problem(
      2.0, 1.0, 0.125, 1.0, 0.1, std::array{0.0, 0.0, 0.0},
      std::array{0.0, 0.0, 0.0}, std::array{0.5, 1.0, 0.0},
      std::array{0.5, -1.0, 0.0}, 2.0, 0.4);
  test_serialization(riemann_problem);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) {
  const double adiabatic_index = 2.0;
  const double left_density = 1.0;
  const double right_density = 0.125;
  const double left_pressure = 1.0;
  const double right_pressure = 0.1;
  const std::array left_velocity{0.0, 0.0, 0.0};
  const std::array right_velocity{0.0, 0.0, 0.0};
  const std::array left_magnetic_field{0.5, 1.0, 0.0};
  const std::array right_magnetic_field{0.5, -1.0, 0.0};
  const double lapse = 2.0;
  const double shift = 0.4;

  const auto member_variables = std::make_tuple(
      adiabatic_index, left_density, right_density, left_pressure,
      right_pressure, left_velocity, right_velocity, left_magnetic_field,
      right_magnetic_field, lapse, shift);

  RiemannProblemProxy riemann_problem(
      adiabatic_index, left_density, right_density, left_pressure,
      right_pressure, left_velocity, right_velocity, left_magnetic_field,
      right_magnetic_field, lapse, shift);

  pypp::check_with_random_values<1>(
      &RiemannProblemProxy::hydro_variables<DataType>, riemann_problem,
      "RiemannProblem",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "compute_pressure", "lorentz_factor", "specific_enthalpy"},
      {{{-1.0, 1.0}}}, member_variables, used_for_size);

  pypp::check_with_random_values<1>(
      &RiemannProblemProxy::grmhd_variables<DataType>, riemann_problem,
      "RiemannProblem",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "compute_pressure", "lorentz_factor", "specific_enthalpy",
       "magnetic_field", "divergence_cleaning_field"},
      {{{-1.0, 1.0}}}, member_variables, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.RiemannProblem",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));
}
