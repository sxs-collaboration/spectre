// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave {
namespace {

using deriv_psi_tag =
    ::Tags::deriv<Tags::Psi, tmpl::size_t<3>, Frame::Inertial>;
using puncture_vars =
    Variables<tmpl::list<Tags::Psi, ::Tags::dt<Tags::Psi>, deriv_psi_tag>>;
// tests that the puncture field corresponds to a circular orbit with angular
// velocity = orbit_radius^(-3/2)
void test_circular_orbit() {
  MAKE_GENERATOR(gen)
  // sample 100 random points around the worldtube
  const double orbit_radius = 7.;
  const double orbit_speed = pow(orbit_radius, -1.5);
  const size_t num_points = 100;
  const tnsr::I<double, 3, Frame::Inertial> wt_coords{{orbit_radius, 0., 0.}};
  std::uniform_real_distribution dist_around_wt(-3., 3.);
  auto sample_points =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&dist_around_wt),
          DataVector(num_points));
  sample_points.get(0) += orbit_radius;
  const double time_0 = 0.;
  const double time_1 = 20.;
  for (size_t order = 0; order <= 2; ++order) {
    CAPTURE(order);
    puncture_vars puncture_t0{num_points};
    Worldtube::puncture_field(make_not_null(&puncture_t0), sample_points,
                              time_0, orbit_radius, 1., order);

    // rotate the sample points and check that the values don't change
    tnsr::I<DataVector, 3, Frame::Inertial> sample_points_rotated(num_points,
                                                                  0.);
    sample_points_rotated.get(0) =
        sample_points.get(0) * cos(orbit_speed * time_1) -
        sample_points.get(1) * sin(orbit_speed * time_1);
    sample_points_rotated.get(1) =
        sample_points.get(0) * sin(orbit_speed * time_1) +
        sample_points.get(1) * cos(orbit_speed * time_1);
    sample_points_rotated.get(2) = sample_points.get(2);
    puncture_vars puncture_t1{num_points};
    Worldtube::puncture_field(make_not_null(&puncture_t1),
                              sample_points_rotated, time_1, orbit_radius, 1.,
                              order);
    Approx local_approx = Approx::custom().epsilon(1.e-11).scale(1.);
    CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::Psi>(puncture_t0).get(),
                                 get<Tags::Psi>(puncture_t1).get(),
                                 local_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(get<::Tags::dt<Tags::Psi>>(puncture_t0).get(),
                                 get<::Tags::dt<Tags::Psi>>(puncture_t1).get(),
                                 local_approx);

    // check that the spatial derivative also gets rotated
    const auto& di_psi = get<deriv_psi_tag>(puncture_t0);
    tnsr::i<DataVector, 3, Frame::Inertial> di_psi_rotated(num_points);
    di_psi_rotated.get(0) = di_psi.get(0) * cos(orbit_speed * time_1) -
                            di_psi.get(1) * sin(orbit_speed * time_1);
    di_psi_rotated.get(1) = di_psi.get(0) * sin(orbit_speed * time_1) +
                            di_psi.get(1) * cos(orbit_speed * time_1);
    di_psi_rotated.get(2) = di_psi.get(2);
    CHECK_ITERABLE_CUSTOM_APPROX(di_psi_rotated,
                                 get<deriv_psi_tag>(puncture_t1), local_approx);
  }
}

// tests the derivative of the puncture field against a finite difference
// calculation
void test_derivative() {
  MAKE_GENERATOR(gen);
  const double time = 17.4;
  const double orbit_radius = 10.;
  for (size_t order = 0; order <= 2; ++order) {
    CAPTURE(order);
    const auto helper_func = [&time, &orbit_radius,
                              &order](const std::array<double, 3>& point) {
      tnsr::I<DataVector, 3, Frame::Inertial> tensor_point(size_t(1));
      tensor_point.get(0) = point.at(0);
      tensor_point.get(1) = point.at(1);
      tensor_point.get(2) = point.at(2);
      puncture_vars singular_field{1};
      Worldtube::puncture_field(make_not_null(&singular_field), tensor_point,
                                time, orbit_radius, 1., order);
      return get<Tags::Psi>(singular_field).get()[0];
    };
    Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
    std::uniform_real_distribution<double> theta_dist{0, M_PI};
    std::uniform_real_distribution<double> phi_dist{0, 2 * M_PI};
    const double wt_radius = 0.1;
    for (size_t i = 0; i < 20; ++i) {
      const auto theta = theta_dist(gen);
      const auto phi = phi_dist(gen);

      const std::array<double, 3> test_point{
          orbit_radius + wt_radius * sin(theta) * cos(phi),
          wt_radius * sin(theta) * sin(phi), wt_radius * cos(theta)};
      const double dx = 1e-4;
      tnsr::I<DataVector, 3, Frame::Inertial> tensor_point(size_t(1));
      tensor_point.get(0) = orbit_radius + wt_radius * sin(theta) * cos(phi);
      tensor_point.get(1) = wt_radius * sin(theta) * sin(phi);
      tensor_point.get(2) = wt_radius * cos(theta);
      puncture_vars singular_field{1};
      Worldtube::puncture_field(make_not_null(&singular_field), tensor_point,
                                time, orbit_radius, 1., order);
      const auto& di_psi = get<deriv_psi_tag>(singular_field);
      for (size_t j = 0; j < 3; ++j) {
        const auto numerical_deriv_j =
            numerical_derivative(helper_func, test_point, j, dx);
        CHECK(di_psi.get(j)[0] == local_approx(numerical_deriv_j));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.PunctureField",
                  "[Unit][Evolution]") {
  test_circular_orbit();
  test_derivative();
}
}  // namespace
}  // namespace CurvedScalarWave
