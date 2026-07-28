// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/KerrSchildDerivatives.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/IterateAccelerationTerms.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube {

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CSW.Worldtube.IterateAccelerationTerms",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/CurvedScalarWave/Worldtube"};
  MAKE_GENERATOR(gen);
  static constexpr size_t Dim = 3;
  const std::uniform_real_distribution<double> dist(-1., 1.);
  const std::uniform_real_distribution<double> pos_dist(2., 10.);
  const std::uniform_real_distribution<double> vel_dist(-0.1, 0.1);
  const gr::Solutions::KerrSchild kerr_schild(1.4, {{0.1, 0.2, 0.3}},
                                              {{0., 0., 0.}});
  const auto pos = make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
      make_not_null(&gen), pos_dist, 1);
  const auto vel = make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
      make_not_null(&gen), vel_dist, 1);

  const auto psi_monopole =
      make_with_random_values<Scalar<double>>(make_not_null(&gen), dist, 1);
  const auto dt_psi_monopole =
      make_with_random_values<Scalar<double>>(make_not_null(&gen), dist, 1);
  const auto psi_dipole = make_with_random_values<tnsr::i<double, Dim>>(
      make_not_null(&gen), dist, 1);
  const auto dt_psi_dipole = make_with_random_values<tnsr::i<double, Dim>>(
      make_not_null(&gen), dist, 1);
  const double charge = 0.1;
  const double mass = 0.85;
  const double time = 10.;
  const double turn_on_time = 20.;
  const double turn_on_interval = 1.;
  const size_t current_iteration = 1;
  auto box = db::create<
      db::AddSimpleTags<
          Tags::AccelerationTerms, Tags::ParticlePositionVelocity<Dim>,
          CurvedScalarWave::Tags::BackgroundSpacetime<
              gr::Solutions::KerrSchild>,
          Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim, Frame::Inertial>,
          Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                               Frame::Inertial>,
          Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Inertial>,
          Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 1, Dim,
                               Frame::Inertial>,
          Tags::Charge, Tags::Mass, ::Tags::Time, Tags::SelfForceTurnOnTime,
          Tags::SelfForceTurnOnInterval, Tags::CurrentIteration>,
      db::AddComputeTags<Tags::BackgroundQuantitiesCompute<Dim>,
                         Tags::GeodesicAccelerationCompute<Dim>>>(
      Scalar<DataVector>{}, std::array<tnsr::I<double, 3>, 2>{pos, vel},
      kerr_schild, psi_monopole, dt_psi_monopole, psi_dipole, dt_psi_dipole,
      charge, std::make_optional(mass), time, std::make_optional(turn_on_time),
      std::make_optional(turn_on_interval), current_iteration);
  db::mutate_apply<IterateAccelerationTerms>(make_not_null(&box));

  const auto& background = db::get<Tags::BackgroundQuantities<Dim>>(box);
  const auto& geodesic_acc = db::get<Tags::GeodesicAcceleration<Dim>>(box);
  const auto& imetric =
      get<gr::Tags::InverseSpacetimeMetric<double, Dim>>(background);
  const auto& metric = get<gr::Tags::SpacetimeMetric<double, Dim>>(background);
  const auto& christoffel =
      get<gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>(background);
  const auto& trace_christoffel =
      get<gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim>>(
          background);
  const auto di_imetric = spatial_derivative_inverse_ks_metric(pos);
  const auto dij_imetric = second_spatial_derivative_inverse_ks_metric(pos);
  const auto di_trace_christoffel =
      spatial_derivative_ks_contracted_christoffel(pos);
  const auto di_metric = spatial_derivative_ks_metric(metric, di_imetric);
  const auto dij_metric = second_spatial_derivative_metric(
      metric, di_metric, di_imetric, dij_imetric);
  const auto di_christoffel = spatial_derivative_christoffel(
      di_metric, dij_metric, imetric, di_imetric);
  const auto& u0 = get(get<Tags::TimeDilationFactor>(background));

  double roll_on = 0.;
  double dt_roll_on = 0.;
  double dt2_roll_on = 0.;

  const auto expected_terms = pypp::call<DataVector>(
      "SelfForce", "iterate_acceleration_terms", vel, psi_monopole,
      dt_psi_monopole, psi_dipole, dt_psi_dipole, roll_on, dt_roll_on,
      dt2_roll_on, imetric, christoffel, trace_christoffel, di_imetric,
      dij_imetric, di_trace_christoffel, di_christoffel, u0, geodesic_acc,
      charge, mass, current_iteration);

  CHECK_ITERABLE_APPROX(expected_terms,
                        get(db::get<Tags::AccelerationTerms>(box)));

  const double new_time = 21.123;
  db::mutate<::Tags::Time>(
      [&new_time](const gsl::not_null<double*> local_time) {
        *local_time = new_time;
      },
      make_not_null(&box));
  db::mutate_apply<IterateAccelerationTerms>(make_not_null(&box));

  const double t_minus_turnup = new_time - turn_on_time;
  roll_on = turn_on_function(t_minus_turnup, turn_on_interval);
  dt_roll_on = dt_turn_on_function(t_minus_turnup, turn_on_interval);
  dt2_roll_on = dt2_turn_on_function(t_minus_turnup, turn_on_interval);

  const auto expected_terms_2 = pypp::call<DataVector>(
      "SelfForce", "iterate_acceleration_terms", vel, psi_monopole,
      dt_psi_monopole, psi_dipole, dt_psi_dipole, roll_on, dt_roll_on,
      dt2_roll_on, imetric, christoffel, trace_christoffel, di_imetric,
      dij_imetric, di_trace_christoffel, di_christoffel, u0, geodesic_acc,
      charge, mass, current_iteration);

  CHECK_ITERABLE_APPROX(expected_terms_2,
                        get(db::get<Tags::AccelerationTerms>(box)));

  static constexpr size_t new_iteration = 7;
  db::mutate<Tags::CurrentIteration>(
      [](const gsl::not_null<size_t*> local_current_iteration) {
        *local_current_iteration = new_iteration;
      },
      make_not_null(&box));
  db::mutate_apply<IterateAccelerationTerms>(make_not_null(&box));

  const auto expected_terms_3 = pypp::call<DataVector>(
      "SelfForce", "iterate_acceleration_terms", vel, psi_monopole,
      dt_psi_monopole, psi_dipole, dt_psi_dipole, roll_on, dt_roll_on,
      dt2_roll_on, imetric, christoffel, trace_christoffel, di_imetric,
      dij_imetric, di_trace_christoffel, di_christoffel, u0, geodesic_acc,
      charge, mass, new_iteration);
  CHECK_ITERABLE_APPROX(expected_terms_3,
                        get(db::get<Tags::AccelerationTerms>(box)));
}
}  // namespace CurvedScalarWave::Worldtube
