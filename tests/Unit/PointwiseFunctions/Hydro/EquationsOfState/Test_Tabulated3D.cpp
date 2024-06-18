// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/Version.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.Tabulated3D",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;

  using TEoS = EoS::Tabulated3D<true>;

  // We do not test proper registration, yet.
  // Will do this once we have the EoS reader setup.
  // For now, we will construct the internal state of
  // the EoS manually.

  // register_derived_classes_with_charm<
  //     EoS::EquationOfState<true, 2>>();
  // register_derived_classes_with_charm<
  //     EoS::EquationOfState<false, 2>>();
  // pypp::SetupLocalPythonEnvironment local_python_env{
  //     "PointwiseFunctions/Hydro/EquationsOfState/"};
  //
  MAKE_GENERATOR(gen);

  std::uniform_int_distribution<size_t> dist_int(3, 5);

  constexpr int Dim = 3;
  constexpr int NumVar = TEoS::NumberOfVars;

  std::array<double, Dim> lower_bounds{{
      -3 * std::log(10),  // TEMP
      -5 * std::log(10),  // RHO
      0.01                // YE
  }};

  std::array<double, Dim> upper_bounds{{1 * std::log(10),   // TEMP
                                        -1 * std::log(10),  // RHO
                                        0.5}};

  // Create data data structures
  std::array<std::vector<double>, Dim> X_data;
  Index<Dim> num_x_points;

  size_t total_num_points = 1;
  for (size_t n = 0; n < Dim; ++n) {
    num_x_points[n] = dist_int(gen);
    total_num_points *= num_x_points[n];

    X_data[n].resize(num_x_points[n]);

    for (size_t m = 0; m < num_x_points[n]; m++) {
      X_data[n][m] =
          lower_bounds[n] + m * (upper_bounds[n] - lower_bounds[n]) /
                                static_cast<double>(num_x_points[n] - 1);
    }
  }
  CAPTURE(num_x_points);
  CAPTURE(X_data);

  // Correct for NumVars
  total_num_points *= TEoS::NumberOfVars;
  CAPTURE(total_num_points);

  // Allocate storage for main table
  std::vector<double> dependent_variables;
  dependent_variables.resize(total_num_points);

  // Fill dependent_variables
  //
  double energy_shift = 0;  // Will test this later

  double eps_min = std::exp(lower_bounds[0]);
  CAPTURE(eps_min);

  if (eps_min < 0) {
    energy_shift = 2. * eps_min;
  }
  CAPTURE(energy_shift);

  auto test_eos = [&](auto state) {
    enum TableIndex { Temp = 0, Rho = 1, Ye = 2 };

    // We use a simple ideal fluid like EOS with a Ye variable Gamma:
    // p = (rho*eps)*Ye = rho T

    std::array<double, TEoS::NumberOfVars> vars;

    // This is not consistent, but better keep this simple
    vars[TEoS::Epsilon] = state[TableIndex::Temp];
    vars[TEoS::Pressure] = state[TableIndex::Temp] + state[TableIndex::Rho];
    vars[TEoS::CsSquared] = state[TableIndex::Ye];

    return vars;
  };

  for (size_t ijk = 0; ijk < total_num_points / NumVar; ++ijk) {
    Index<Dim> index;
    auto tmp = std::array<double, Dim>{};

    // Uncompress index
    size_t myind = ijk;
    for (size_t nn = 0; nn < Dim - 1; ++nn) {
      index[nn] = myind % num_x_points[nn];
      myind = (myind - index[nn]) / num_x_points[nn];
      tmp[nn] = X_data[nn][index[nn]];
    }
    index[Dim - 1] = myind;
    tmp[Dim - 1] = X_data[Dim - 1][index[Dim - 1]];

    for (size_t nv = 0; nv < NumVar; ++nv) {
      std::array<double, NumVar> Fx = test_eos(tmp);
      dependent_variables[nv + NumVar * ijk] = Fx[nv];
    }
  }

  // Construct EOS

  double enthalpy_minimum = 1.;

  TEoS eos = TEoS(X_data[2], X_data[1], X_data[0], dependent_variables,
                  energy_shift, enthalpy_minimum);

  CHECK(std::abs(std::exp(lower_bounds[0]) - eos.temperature_lower_bound()) <
        1.e-12);
  CHECK(std::abs(std::exp(lower_bounds[1]) -
                 eos.rest_mass_density_lower_bound()) < 1.e-12);
  CHECK(std::abs((lower_bounds[2]) - eos.electron_fraction_lower_bound()) <
        1.e-12);

  CHECK(std::abs(std::exp(upper_bounds[0]) - eos.temperature_upper_bound()) <
        1.e-12);
  CHECK(std::abs(std::exp(upper_bounds[1]) -
                 eos.rest_mass_density_upper_bound()) < 1.e-12);
  CHECK(std::abs((upper_bounds[2]) - eos.electron_fraction_upper_bound()) <
        1.e-12);

  // Construct a test state
  std::array<double, 3> pure_state{{1., 1.e-3, 0.3}};

  std::array<Scalar<double>, 3> state{};
  std::array<Scalar<DataVector>, 3> vector_state{};

  constexpr size_t data_vector_length = 5;

  for (size_t n = 0; n < 3; ++n) {
    get(gsl::at(state, n)) = gsl::at(pure_state, n);
    get(gsl::at(vector_state, n)) =
        DataVector{data_vector_length, gsl::at(pure_state, n)};
  }

  pure_state[0] = std::log(pure_state[0]);
  pure_state[1] = std::log(pure_state[1]);
  CAPTURE(pure_state);
  CAPTURE(vector_state);

  const auto output = test_eos(pure_state);
  CAPTURE(output);

  CHECK(std::abs((std::exp(output[TEoS::Epsilon]) + energy_shift) -
                 get(eos.specific_internal_energy_from_density_and_temperature(
                     state[1], state[0], state[2]))) < 1.e-12);
  CHECK(std::abs((std::exp(output[TEoS::Pressure])) -
                 get(eos.pressure_from_density_and_temperature(
                     state[1], state[0], state[2]))) < 1.e-12);
  CHECK(std::abs(output[TEoS::CsSquared]) -
            get(eos.sound_speed_squared_from_density_and_temperature(
                state[1], state[0], state[2])) <
        1.e-12);
  CHECK(not eos.is_barotropic());
  CHECK(not eos.is_equilibrium());

  const auto eps_interp =
      eos.specific_internal_energy_from_density_and_temperature(
          vector_state[1], vector_state[0], vector_state[2]);
  CAPTURE(eps_interp);

  // Ensure the tabulated EoS is applied to all elements in the datavector, not
  // just the first element
  for (size_t vector_index = 0; vector_index < data_vector_length;
       vector_index++) {
    CHECK(std::abs(std::exp(pure_state[0]) -
                   get(eos.temperature_from_density_and_energy(
                       vector_state[1], eps_interp,
                       vector_state[2]))[vector_index]) < 1.e-12);
  }

  CHECK(std::abs((std::exp(output[TEoS::Epsilon]) + energy_shift) -
                 get(eos.specific_internal_energy_from_density_and_temperature(
                     vector_state[1], vector_state[0], vector_state[2]))[0]) <
        1.e-12);
  CHECK(std::abs((std::exp(output[TEoS::Pressure])) -
                 get(eos.pressure_from_density_and_temperature(
                     vector_state[1], vector_state[0], vector_state[2]))[0]) <
        1.e-12);
  CHECK(std::abs(output[TEoS::CsSquared] -
                 get(eos.sound_speed_squared_from_density_and_temperature(
                     vector_state[1], vector_state[0], vector_state[2]))[0]) <
        1.e-12);

  const auto eps_interp_vector =
      eos.specific_internal_energy_from_density_and_temperature(
          vector_state[1], vector_state[0], vector_state[2]);
  CAPTURE(eps_interp_vector);

  CHECK(std::abs(std::exp(pure_state[0]) -
                 get(eos.temperature_from_density_and_energy(
                     vector_state[1], eps_interp_vector, vector_state[2]))[0]) <
        1.e-12);

  auto test_against_reference_values = [&](auto& this_eos) {
    get(state[1]) = 1.e-4;

    CHECK_ITERABLE_APPROX(
        get(this_eos.specific_internal_energy_from_density_and_temperature(
            state[1], state[0], state[2])),
        0.30204636358732767);
    CHECK_ITERABLE_APPROX(get(this_eos.pressure_from_density_and_temperature(
                              state[1], state[0], state[2])),
                          0.00001103280164124);
    CHECK_ITERABLE_APPROX(
        get(this_eos.sound_speed_squared_from_density_and_temperature(
            state[1], state[0], state[2])),
        0.41669901507784435);
  };

  // Test against reference values

  std::string h5_file_name{
      unit_test_src_path() +
      "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5"};

  h5::H5File<h5::AccessType::ReadOnly> eos_file{h5_file_name};
  const auto& compose_eos = eos_file.get<h5::EosTable>("/dd2");

  eos.initialize(compose_eos);

  test_against_reference_values(eos);

  // Test serialization

  register_derived_classes_with_charm<EoS::EquationOfState<true, 3>>();
  const auto eos_pointer =
      serialize_and_deserialize(TestHelpers::test_creation<
                                std::unique_ptr<EoS::EquationOfState<true, 3>>>(
          {"Tabulated3D:\n"
           "  TableFilename: " +
           unit_test_src_path() +
           "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5\n"
           "  TableSubFilename: 'dd2'"}));
  const EoS::Tabulated3D<true>& deserialized_eos =
      dynamic_cast<const EoS::Tabulated3D<true>&>(*eos_pointer);
  TestHelpers::EquationsOfState::test_get_clone(deserialized_eos);

  CHECK(deserialized_eos == eos);

  test_against_reference_values(deserialized_eos);
}
