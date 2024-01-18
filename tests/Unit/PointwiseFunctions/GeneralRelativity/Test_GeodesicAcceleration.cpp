// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicAcceleration.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {

template <typename DataType>
void test_circular_orbit_kerr_schild() {
  MAKE_GENERATOR(gen);
  const double mass = 3.2;
  const std::array<double, 3> spin{{0., 0., 0.}};
  const std::array<double, 3> center{{0., 0., 0.}};
  gr::Solutions::KerrSchild kerr_schild(mass, spin, center);
  const auto used_for_size = Scalar<DataType>(static_cast<size_t>(100));
  std::uniform_real_distribution<> radius_dist(7., 100.);
  std::uniform_real_distribution<> angle_dist(0., 2. * M_PI);
  const auto radius = make_with_random_values<DataType>(
      make_not_null(&gen), radius_dist, used_for_size);
  const auto angle = make_with_random_values<DataType>(
      make_not_null(&gen), angle_dist, used_for_size);
  const auto angular_velocity = sqrt(mass) / (radius * sqrt(radius));

  const tnsr::I<DataType, 3> position{
      {radius * cos(angle), radius * sin(angle),
       make_with_value<DataType>(used_for_size, 0.)}};
  const tnsr::I<DataType, 3> velocity{
      {-radius * angular_velocity * sin(angle),
       radius * angular_velocity * cos(angle),
       make_with_value<DataType>(used_for_size, 0.)}};
  const tnsr::I<DataType, 3> expected_acceleration{
      {-radius * square(angular_velocity) * cos(angle),
       -radius * square(angular_velocity) * sin(angle),
       make_with_value<DataType>(used_for_size, 0.)}};

  const auto christoffel = get<
      gr::Tags::SpacetimeChristoffelSecondKind<DataType, 3, Frame::Inertial>>(
      kerr_schild.variables(position, 0.,
                            tmpl::list<gr::Tags::SpacetimeChristoffelSecondKind<
                                DataType, 3, Frame::Inertial>>{}));

  const auto geodesic_acc = gr::geodesic_acceleration(velocity, christoffel);
  CHECK_ITERABLE_APPROX(geodesic_acc, expected_acceleration);
}

std::array<tnsr::I<double, 3>, 2> state_to_tnsr(
    const std::array<double, 6>& state) {
  tnsr::I<double, 3> pos_tensor{{state[0], state[1], state[2]}};
  tnsr::I<double, 3> vel_tensor{{state[3], state[4], state[5]}};
  return {std::move(pos_tensor), std::move(vel_tensor)};
}

// struct with interface conforming to a `System` as used in
// boost::numeric::odeint
struct BoostGeodesicIntegrator {
  gr::Solutions::KerrSchild kerr_schild;
  void operator()(const std::array<double, 6>& state,
                  std::array<double, 6>& derivative_of_state,
                  const double /*time*/) const {
    const auto [position, velocity] = state_to_tnsr(state);
    const auto christoffel = get<
        gr::Tags::SpacetimeChristoffelSecondKind<double, 3, Frame::Inertial>>(
        kerr_schild.variables(
            position, 0.,
            tmpl::list<gr::Tags::SpacetimeChristoffelSecondKind<
                double, 3, Frame::Inertial>>{}));
    const auto geodesic_acc = gr::geodesic_acceleration(velocity, christoffel);
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(derivative_of_state, i) = state.at(i + 3);
      gsl::at(derivative_of_state, i + 3) = geodesic_acc.get(i);
    }
  }
};

// struct with interface conforming to an `Observer` as used in
// boost::numeric::odeint
struct BoostObserver {
  std::vector<std::array<double, 6>>& states;
  std::vector<double>& times;
  void operator()(const std::array<double, 6>& state, double t) {
    states.push_back(state);
    times.push_back(t);
  }
};

tnsr::a<double, 3> lower_four_velocity(
    const std::array<double, 6>& state,
    const gr::Solutions::KerrSchild& kerr_schild) {
  const auto [pos, vel] = state_to_tnsr(state);
  const auto spacetime_vars = kerr_schild.variables(
      pos, 0.,
      tmpl::list<gr::Tags::Lapse<double>,
                 gr::Tags::Shift<double, 3, Frame::Inertial>,
                 gr::Tags::SpatialMetric<double, 3, Frame::Inertial>>{});
  const auto spacetime_metric = gr::spacetime_metric(
      get<gr::Tags::Lapse<double>>(spacetime_vars),
      get<gr::Tags::Shift<double, 3, Frame::Inertial>>(spacetime_vars),
      get<gr::Tags::SpatialMetric<double, 3, Frame::Inertial>>(spacetime_vars));
  double temp = spacetime_metric.get(0, 0);
  for (size_t i = 0; i < 3; ++i) {
    temp += 2. * spacetime_metric.get(i + 1, 0) * vel.get(i);
    for (size_t j = 0; j < 3; ++j) {
      temp += spacetime_metric.get(i + 1, j + 1) * vel.get(i) * vel.get(j);
    }
  }
  // the expression for u0 follows directly from u^i = u^0 * v^i and the
  // normalization of the four velocity
  const double u0 = sqrt(-1. / temp);
  const tnsr::A<double, 3> u{
      {u0, vel.get(0) * u0, vel.get(1) * u0, vel.get(2) * u0}};
  return tenex::evaluate<ti::a>(spacetime_metric(ti::a, ti::b) * u(ti::B));
}

// checks that the conserved quantities due to the spacetime symmetries i.e.
// Killing vectors are conserved during geodesic evolution
void test_conserved_quantities_kerr_schild() {
  const double mass = 1.3;
  const std::array<double, 3> spin{{0., 0., 0.3}};
  const std::array<double, 3> center{{0., 0., 0.}};
  gr::Solutions::KerrSchild kerr_schild(mass, spin, center);

  const double t_max = 1000.0;

  // corresponds to a strongly perturbed circular orbit
  std::array<double, 6> initial_state = {
      10., 1.0, 0.8, 0.1, 1. / sqrt(10.) * 1.2, 0.3};

  std::vector<std::array<double, 6>> states{};
  std::vector<double> times{};
  BoostObserver observer{states, times};
  boost::numeric::odeint::integrate_adaptive(
      boost::numeric::odeint::make_controlled(
          1e-15, 1e-15,
          boost::numeric::odeint::runge_kutta_cash_karp54<
              std::array<double, 6>>()),
      BoostGeodesicIntegrator{kerr_schild}, initial_state, 0.0, t_max, 1e-5,
      observer);

  const auto initial_four_velocity =
      lower_four_velocity(initial_state, kerr_schild);

  // the two Killing vectors in Kerr-Schild coordinates
  const tnsr::A<double, 3> timelike_killing{{1., 0., 0., 0.}};
  const tnsr::A<double, 3> azimuthal_killing_initial{
      {0., initial_state[1], -initial_state[0], 0.}};

  const double initial_energy =
      get(dot_product(initial_four_velocity, timelike_killing));
  const double initial_angular_momentum =
      get(dot_product(initial_four_velocity, azimuthal_killing_initial));

  for (const auto& state : states) {
    const auto four_velocity = lower_four_velocity(state, kerr_schild);
    const double energy = get(dot_product(four_velocity, timelike_killing));
    const tnsr::A<double, 3> azimuthal_killing{{0., state[1], -state[0], 0.}};
    const double angular_momentum =
        get(dot_product(four_velocity, azimuthal_killing));
    CHECK(initial_energy == approx(energy));
    CHECK(initial_angular_momentum == approx(angular_momentum));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.GeodesicAcceleration",
    "[Unit][PointwiseFunctions]") {
  test_circular_orbit_kerr_schild<double>();
  test_circular_orbit_kerr_schild<DataVector>();
  test_conserved_quantities_kerr_schild();
}
