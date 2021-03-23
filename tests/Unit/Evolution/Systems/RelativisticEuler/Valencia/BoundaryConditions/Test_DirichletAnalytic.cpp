// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/System.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/SmoothFlow.hpp"
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
  using packed_container = RelativisticEuler::Solutions::SmoothFlow<Dim>;
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

  static inline unpacked_container unpack(
      const packed_container& /*packed*/,
      const size_t /*grid_point_index*/) noexcept {
    // No way of getting the args from the boundary condition.
    return Dim;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = create_container();
  }

  static inline size_t get_size(const packed_container& /*packed*/) noexcept {
    return 1;
  }
};

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time,
      Tags::AnalyticSolution<RelativisticEuler::Solutions::SmoothFlow<Dim>>>>(
      0.5, ConvertSmoothFlow<Dim>::create_container());

  helpers::test_boundary_condition_with_python<
      RelativisticEuler::Valencia::BoundaryConditions::DirichletAnalytic<Dim>,
      RelativisticEuler::Valencia::BoundaryConditions::BoundaryCondition<Dim>,
      RelativisticEuler::Valencia::System<
          Dim, typename RelativisticEuler::Solutions::SmoothFlow<
                   Dim>::equation_of_state_type>,
      tmpl::list<
          RelativisticEuler::Valencia::BoundaryCorrections::Rusanov<Dim>>,
      tmpl::list<ConvertSmoothFlow<Dim>>>(
      make_not_null(&gen),
      "Evolution.Systems.RelativisticEuler.Valencia.BoundaryConditions."
      "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              RelativisticEuler::Valencia::Tags::TildeD>,
          helpers::Tags::PythonFunctionName<
              RelativisticEuler::Valencia::Tags::TildeTau>,
          helpers::Tags::PythonFunctionName<
              RelativisticEuler::Valencia::Tags::TildeS<Dim>>,

          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<RelativisticEuler::Valencia::Tags::TildeD,
                           tmpl::size_t<Dim>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<RelativisticEuler::Valencia::Tags::TildeTau,
                           tmpl::size_t<Dim>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<RelativisticEuler::Valencia::Tags::TildeS<Dim>,
                           tmpl::size_t<Dim>, Frame::Inertial>>,

          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<Dim>>,
          helpers::Tags::PythonFunctionName<gr::Tags::SpatialMetric<Dim>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::RestMassDensity<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::SpecificInternalEnergy<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::SpecificEnthalpy<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::SpatialVelocity<DataVector, Dim>>>{
          "soln_error", "soln_tilde_d", "soln_tilde_tau", "soln_tilde_s",
          "soln_flux_tilde_d", "soln_flux_tilde_tau", "soln_flux_tilde_s",
          "soln_lapse", "soln_shift", "soln_spatial_metric",
          "soln_rest_mass_density", "soln_specific_internal_energy",
          "soln_specific_enthalpy", "soln_spatial_velocity"},
      "DirichletAnalytic:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
      box_analytic_soln, tuples::TaggedTuple<>{});

  // Note: Currently there is no analytic data for the
  // RelativisticEuler::Valencia system. When that is implemented, a test should
  // be added.
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.RelativisticEuler.Valencia.BoundaryConditions.DirichletAnalytic",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
}
