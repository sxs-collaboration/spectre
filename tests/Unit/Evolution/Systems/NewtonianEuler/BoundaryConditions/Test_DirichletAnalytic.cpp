// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/KhInstability.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
template <size_t Dim>
struct ConvertSmoothFlow {
  using unpacked_container = int;
  using packed_container = NewtonianEuler::Solutions::SmoothFlow<Dim>;
  using packed_type = double;

  static packed_container create_container() {
    std::array<double, Dim> wave_vector{};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(wave_vector, i) = 0.1 + i;
    }
    std::array<double, Dim> mean_velocity{};
    for (size_t i = 0; i < Dim; ++i) {
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
    return Dim;
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

template <size_t Dim>
struct ConvertKhInstability {
  using unpacked_container = int;
  using packed_container = NewtonianEuler::AnalyticData::KhInstability<Dim>;
  using packed_type = double;

  static packed_container create_container() {
    const double adiabatic_index = 1.4;
    const double strip_bimedian_height = 0.5;
    const double strip_thickness = 0.5;
    const double strip_density = 2.0;
    const double strip_velocity = 0.5;
    const double background_density = 1.0;
    const double background_velocity = -0.5;
    const double pressure = 2.5;
    const double perturb_amplitude = 0.1;
    const double perturb_width = 0.03;
    return {adiabatic_index,     strip_bimedian_height,
            strip_thickness,     strip_density,
            strip_velocity,      background_density,
            background_velocity, pressure,
            perturb_amplitude,   perturb_width};
  }

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    // No way of getting the args from the boundary condition.
    return Dim;
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

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time,
      Tags::AnalyticSolution<NewtonianEuler::Solutions::SmoothFlow<Dim>>>>(
      0.5, ConvertSmoothFlow<Dim>::create_container());

  helpers::test_boundary_condition_with_python<
      NewtonianEuler::BoundaryConditions::DirichletAnalytic<Dim>,
      NewtonianEuler::BoundaryConditions::BoundaryCondition<Dim>,
      NewtonianEuler::System<Dim,
                             NewtonianEuler::Solutions::SmoothFlow<Dim>>,
      tmpl::list<NewtonianEuler::BoundaryCorrections::Rusanov<Dim>>,
      tmpl::list<ConvertSmoothFlow<Dim>>>(
      make_not_null(&gen),
      "Evolution.Systems.NewtonianEuler.BoundaryConditions.DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::MassDensityCons>,
          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::MomentumDensity<Dim>>,
          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::EnergyDensity>,

          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<NewtonianEuler::Tags::MassDensityCons,
                           tmpl::size_t<Dim>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<NewtonianEuler::Tags::MomentumDensity<Dim>,
                           tmpl::size_t<Dim>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<NewtonianEuler::Tags::EnergyDensity,
                           tmpl::size_t<Dim>, Frame::Inertial>>,

          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::Velocity<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<
              NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>>{
          "soln_error", "soln_mass_density", "soln_momentum_density",
          "soln_energy_density", "soln_flux_mass_density",
          "soln_flux_momentum_density", "soln_flux_energy_density",
          "soln_velocity", "soln_specific_internal_energy"},
      "DirichletAnalytic:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
      box_analytic_soln, tuples::TaggedTuple<>{});

  if constexpr (Dim > 1) {
    // KH-instability is only implemented in 2d and 3d. If/when we have 1d
    // analytic data that should also be tested.
    const auto box_analytic_data = db::create<db::AddSimpleTags<
        Tags::Time,
        Tags::AnalyticData<NewtonianEuler::AnalyticData::KhInstability<Dim>>>>(
        0.5, ConvertKhInstability<Dim>::create_container());

    helpers::test_boundary_condition_with_python<
        NewtonianEuler::BoundaryConditions::DirichletAnalytic<Dim>,
        NewtonianEuler::BoundaryConditions::BoundaryCondition<Dim>,
        NewtonianEuler::System<
            Dim,
            NewtonianEuler::AnalyticData::KhInstability<Dim>>,
        tmpl::list<NewtonianEuler::BoundaryCorrections::Rusanov<Dim>>,
        tmpl::list<ConvertKhInstability<Dim>>>(
        make_not_null(&gen),
        "Evolution.Systems.NewtonianEuler.BoundaryConditions.DirichletAnalytic",
        tuples::TaggedTuple<
            helpers::Tags::PythonFunctionForErrorMessage<>,
            helpers::Tags::PythonFunctionName<
                NewtonianEuler::Tags::MassDensityCons>,
            helpers::Tags::PythonFunctionName<
                NewtonianEuler::Tags::MomentumDensity<Dim>>,
            helpers::Tags::PythonFunctionName<
                NewtonianEuler::Tags::EnergyDensity>,

            helpers::Tags::PythonFunctionName<
                ::Tags::Flux<NewtonianEuler::Tags::MassDensityCons,
                             tmpl::size_t<Dim>, Frame::Inertial>>,
            helpers::Tags::PythonFunctionName<
                ::Tags::Flux<NewtonianEuler::Tags::MomentumDensity<Dim>,
                             tmpl::size_t<Dim>, Frame::Inertial>>,
            helpers::Tags::PythonFunctionName<
                ::Tags::Flux<NewtonianEuler::Tags::EnergyDensity,
                             tmpl::size_t<Dim>, Frame::Inertial>>,

            helpers::Tags::PythonFunctionName<
                NewtonianEuler::Tags::Velocity<DataVector, Dim>>,
            helpers::Tags::PythonFunctionName<
                NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>>{
            "data_error", "data_mass_density", "data_momentum_density",
            "data_energy_density", "data_flux_mass_density",
            "data_flux_momentum_density", "data_flux_energy_density",
            "data_velocity", "data_specific_internal_energy"},
        "DirichletAnalytic:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
        box_analytic_data, tuples::TaggedTuple<>{});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianEuler.BoundaryConditions.DirichletAnalytic",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
}
