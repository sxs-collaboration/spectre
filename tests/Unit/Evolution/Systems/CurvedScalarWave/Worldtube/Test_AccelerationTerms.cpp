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
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicAcceleration.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave {
namespace {

using deriv_psi_tag =
    ::Tags::deriv<Tags::Psi, tmpl::size_t<3>, Frame::Inertial>;
using puncture_vars =
    Variables<tmpl::list<Tags::Psi, ::Tags::dt<Tags::Psi>, deriv_psi_tag>>;

// tests the derivative of the puncture field against a finite difference
// calculation
void test_derivative() {
  MAKE_GENERATOR(gen);
  Approx local_approx = Approx::custom().epsilon(1e-12).scale(1.0);
  std::uniform_real_distribution<double> theta_dist{0, M_PI};
  std::uniform_real_distribution<double> phi_dist{0, 2 * M_PI};
  std::uniform_real_distribution<double> pos_dist{2., 10.};
  std::uniform_real_distribution<double> vel_acc_dist{-0.1, 0.1};

  const double wt_radius = 0.1;
  for (size_t order = 0; order <= 0; ++order) {
    CAPTURE(order);
    const auto random_position =
        make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
            make_not_null(&gen), make_not_null(&pos_dist), 1);
    const auto random_velocity =
        make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
            make_not_null(&gen), make_not_null(&vel_acc_dist), 1);
    const auto random_acceleration =
        make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
            make_not_null(&gen), make_not_null(&vel_acc_dist), 1);
    const auto random_self_force = make_with_random_values<Scalar<DataVector>>(
        make_not_null(&gen), make_not_null(&vel_acc_dist), DataVector(6));
    const size_t size = 1;
    const auto helper_func = [&random_position, &random_velocity,
                              &random_acceleration, &random_self_force,
                              &size](const std::array<double, 3>& point) {
      tnsr::I<DataVector, 3, Frame::Inertial> tensor_point(size);
      tensor_point.get(0) = point.at(0);
      tensor_point.get(1) = point.at(1);
      tensor_point.get(2) = point.at(2);
      puncture_vars acc_terms{1};
      Worldtube::acceleration_terms_0(
          make_not_null(&acc_terms), tensor_point, random_position,
          random_velocity, random_acceleration, 1., get(random_self_force)[0],
          get(random_self_force)[1], get(random_self_force)[2],
          get(random_self_force)[3], get(random_self_force)[4],
          get(random_self_force)[5]);
      return get<Tags::Psi>(acc_terms).get()[0];
    };
    for (size_t i = 0; i < 20; ++i) {
      const auto theta = theta_dist(gen);
      const auto phi = phi_dist(gen);
      std::array<double, 3> test_point{wt_radius * sin(theta) * cos(phi),
                                       wt_radius * sin(theta) * sin(phi),
                                       wt_radius * cos(theta)};
      tnsr::I<DataVector, 3, Frame::Inertial> tensor_point(size);
      for (size_t j = 0; j < 3; ++j) {
        gsl::at(test_point, j) -= random_position.get(j);
        tensor_point.get(j)[0] = test_point.at(j);
      }
      const double dx = 1e-4;
      puncture_vars acc_terms{1};
      Worldtube::acceleration_terms_0(
          make_not_null(&acc_terms), tensor_point, random_position,
          random_velocity, random_acceleration, 1., get(random_self_force)[0],
          get(random_self_force)[1], get(random_self_force)[2],
          get(random_self_force)[3], get(random_self_force)[4],
          get(random_self_force)[5]);
      const auto& di_psi = get<deriv_psi_tag>(acc_terms);
      for (size_t j = 0; j < 3; ++j) {
        const auto numerical_deriv_j =
            numerical_derivative(helper_func, test_point, j, dx);
        CHECK(di_psi.get(j)[0] == local_approx(numerical_deriv_j));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.AccelerationTerms",
                  "[Unit][Evolution]") {
  test_derivative();
}
}  // namespace
}  // namespace CurvedScalarWave
