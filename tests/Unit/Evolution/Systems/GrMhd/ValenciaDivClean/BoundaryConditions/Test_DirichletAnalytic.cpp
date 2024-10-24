// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticRotor.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition,
            tmpl::list<grmhd::ValenciaDivClean::BoundaryConditions::
                           DirichletAnalytic>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   grmhd::ValenciaDivClean::InitialData::initial_data_list>,
        tmpl::pair<
            grmhd::AnalyticData::InitialMagneticFields::InitialMagneticField,
            tmpl::list<grmhd::AnalyticData::InitialMagneticFields::Poloidal>>>;
  };
};
}  // namespace

namespace {
struct ConvertSmoothFlow {
  using unpacked_container = int;
  using packed_container = grmhd::Solutions::SmoothFlow;
  using packed_type = double;

  static packed_container create_container() {
    constexpr size_t dim = 3;
    std::array<double, dim> wave_vector{};
    for (size_t i = 0; i < dim; ++i) {
      gsl::at(wave_vector, i) = 0.1 + i;
    }
    std::array<double, dim> mean_velocity{};
    for (size_t i = 0; i < dim; ++i) {
      gsl::at(mean_velocity, i) = 0.9 - i * 0.5;
    }
    const double pressure = 1.0;
    const double adiabatic_index = 5.0 / 3.0;
    const double perturbation_size = 0.2;
    return {mean_velocity, wave_vector, pressure, adiabatic_index,
            perturbation_size};
  }

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    // No way of getting the args from the boundary condition.
    return 3;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = create_container();
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

struct ConvertMagneticRotor {
  using unpacked_container = int;
  using packed_container = grmhd::AnalyticData::MagneticRotor;
  using packed_type = double;

  static packed_container create_container() {
    const double rotor_radius = 0.1;
    const double rotor_density = 10.0;
    const double background_density = 1.0;
    const double pressure = 1.0;
    const double angular_velocity = 9.95;
    const std::array<double, 3> magnetic_field{{3.54490770181103205, 0.0, 0.0}};
    const double adiabatic_index = 5.0 / 3.0;
    return {rotor_radius,     rotor_density,  background_density, pressure,
            angular_velocity, magnetic_field, adiabatic_index};
  }

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    // No way of getting the args from the boundary condition.
    return 3;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = create_container();
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

void test_soln() {
  register_classes_with_charm(
      grmhd::ValenciaDivClean::InitialData::initial_data_list{});
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticSolution<grmhd::Solutions::SmoothFlow>>>(
      0.5, ConvertSmoothFlow::create_container());

  helpers::test_boundary_condition_with_python<
      grmhd::ValenciaDivClean::BoundaryConditions::DirichletAnalytic,
      grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition,
      grmhd::ValenciaDivClean::System,
      tmpl::list<grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov>,
      tmpl::list<ConvertSmoothFlow>,
      tmpl::list<Tags::AnalyticSolution<grmhd::Solutions::SmoothFlow>>,
      Metavariables>(
      make_not_null(&gen),
      "Evolution.Systems.GrMhd.ValenciaDivClean.BoundaryConditions."
      "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeD>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeYe>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeTau>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildePhi>,

          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                           tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeYe,
                           tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau,
                           tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>,
              tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>,
              tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi,
                           tmpl::size_t<3>, Frame::Inertial>>,

          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<DataVector, 3>>,
          helpers::Tags::PythonFunctionName<hydro::Tags::SpatialVelocityOneForm<
              DataVector, 3, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::RestMassDensity<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::ElectronFraction<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::Temperature<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>>{
          "soln_error", "soln_tilde_d", "soln_tilde_ye", "soln_tilde_tau",
          "soln_tilde_s", "soln_tilde_b", "soln_tilde_phi", "soln_flux_tilde_d",
          "soln_flux_tilde_ye", "soln_flux_tilde_tau", "soln_flux_tilde_s",
          "soln_flux_tilde_b", "soln_flux_tilde_phi", "soln_lapse",
          "soln_shift",
          "soln_spatial_velocity",  // flat spacetime, so up and down indices
                                    // are same
          "soln_rest_mass_density", "soln_electron_fraction_long",
          "soln_temperature_long", "soln_spatial_velocity"},
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
      "    SmoothFlow:\n"
      "      WaveVector: [0.1, 1.1, 2.1 ]\n"
      "      MeanVelocity: [0.9, 0.4, -0.1]\n"
      "      Pressure: 1.0\n"
      "      AdiabaticIndex: 1.6666666666666666\n"
      "      PerturbationSize: 0.2",
      Index<2>{5}, box_analytic_soln, tuples::TaggedTuple<>{});
}

void test_data() {
  register_classes_with_charm(
      grmhd::ValenciaDivClean::InitialData::initial_data_list{});
  MAKE_GENERATOR(gen);
  const auto box_analytic_data = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticData<grmhd::AnalyticData::MagneticRotor>>>(
      0.5, ConvertMagneticRotor::create_container());

  helpers::test_boundary_condition_with_python<
      grmhd::ValenciaDivClean::BoundaryConditions::DirichletAnalytic,
      grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition,
      grmhd::ValenciaDivClean::System,
      tmpl::list<grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov>,
      tmpl::list<ConvertMagneticRotor>,
      tmpl::list<Tags::AnalyticData<grmhd::AnalyticData::MagneticRotor>>,
      Metavariables>(
      make_not_null(&gen),
      "Evolution.Systems.GrMhd.ValenciaDivClean.BoundaryConditions."
      "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeD>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeYe>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeTau>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildePhi>,

          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                           tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeYe,
                           tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau,
                           tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>,
              tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>,
              tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi,
                           tmpl::size_t<3>, Frame::Inertial>>,

          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<DataVector, 3>>,
          helpers::Tags::PythonFunctionName<hydro::Tags::SpatialVelocityOneForm<
              DataVector, 3, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::RestMassDensity<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::ElectronFraction<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::Temperature<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>>{
          "soln_error", "data_tilde_d", "data_tilde_ye", "data_tilde_tau",
          "data_tilde_s", "data_tilde_b", "data_tilde_phi", "data_flux_tilde_d",
          "data_flux_tilde_ye", "data_flux_tilde_tau", "data_flux_tilde_s",
          "data_flux_tilde_b", "data_flux_tilde_phi", "soln_lapse",
          "soln_shift",
          "data_spatial_velocity",  // flat spacetime, so up and down indices
                                    // are same
          "data_rest_mass_density", "data_electron_fraction_long",
          "data_temperature", "data_spatial_velocity"},
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
      "    MagneticRotor:\n"
      "      RotorRadius: 0.1\n"
      "      RotorDensity: 10.0\n"
      "      BackgroundDensity: 1.0\n"
      "      Pressure: 1.0\n"
      "      AngularVelocity: 9.95\n"
      "      MagneticField: [3.54490770181103205, 0.0, 0.0]\n"
      "      AdiabaticIndex: 1.666666666666666666",
      Index<2>{5}, box_analytic_data, tuples::TaggedTuple<>{});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GrMhd.ValenciaDivClean.BoundaryConditions.DirichletAnalytic",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test_soln();
  test_data();
}
