// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/Robin.hpp"
#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::BoundaryConditions {

namespace {

const std::string py_module{"Elliptic.Systems.Xcts.BoundaryConditions.Robin"};

tnsr::i<DataVector, 3> make_spherical_face_normal_flat_cartesian(
    tnsr::I<DataVector, 3> x, const std::array<double, 3>& center) {
  for (size_t d = 0; d < 3; ++d) {
    x.get(d) -= gsl::at(center, d);
  }
  Scalar<DataVector> euclidean_radius = magnitude(x);
  tnsr::i<DataVector, 3> face_normal{x.begin()->size()};
  get<0>(face_normal) = -get<0>(x) / get(euclidean_radius);
  get<1>(face_normal) = -get<1>(x) / get(euclidean_radius);
  get<2>(face_normal) = -get<2>(x) / get(euclidean_radius);
  return face_normal;
}

template <Xcts::Equations EnabledEquations, bool Linearized>
void test_robin(const Robin<EnabledEquations>& boundary_condition) {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1., 1.);
  const size_t num_points = 3;
  const auto direction = Direction<3>::upper_zeta();
  const std::array<double, 3> center{{0., 0., 0.}};
  const auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&gen), make_not_null(&dist), num_points);
  const auto face_normal = make_spherical_face_normal_flat_cartesian(x, center);
  const auto box = db::create<db::AddSimpleTags<
      domain::Tags::Faces<3, domain::Tags::Coordinates<3, Frame::Inertial>>,
      domain::Tags::Faces<3, domain::Tags::FaceNormal<3, Frame::Inertial>>>>(
      DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, x}},
      DirectionMap<3, tnsr::i<DataVector, 3>>{{direction, face_normal}});
  auto conformal_factor_minus_one = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&dist), num_points);
  Scalar<DataVector> n_dot_conformal_factor_gradient{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  tnsr::i<DataVector, 3> deriv_conformal_factor{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  if constexpr (EnabledEquations == Xcts::Equations::Hamiltonian) {
    elliptic::apply_boundary_condition<Linearized, void,
                                       tmpl::list<Robin<EnabledEquations>>>(
        boundary_condition, box, direction,
        make_not_null(&conformal_factor_minus_one),
        make_not_null(&n_dot_conformal_factor_gradient),
        deriv_conformal_factor);
  } else {
    auto lapse_times_conformal_factor_minus_one =
        make_with_random_values<Scalar<DataVector>>(
            make_not_null(&gen), make_not_null(&dist), num_points);
    Scalar<DataVector> n_dot_lapse_times_conformal_factor_gradient{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    tnsr::i<DataVector, 3> deriv_lapse_times_conformal_factor{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    if constexpr (EnabledEquations == Xcts::Equations::HamiltonianAndLapse) {
      elliptic::apply_boundary_condition<Linearized, void,
                                         tmpl::list<Robin<EnabledEquations>>>(
          boundary_condition, box, direction,
          make_not_null(&conformal_factor_minus_one),
          make_not_null(&lapse_times_conformal_factor_minus_one),
          make_not_null(&n_dot_conformal_factor_gradient),
          make_not_null(&n_dot_lapse_times_conformal_factor_gradient),
          deriv_conformal_factor, deriv_lapse_times_conformal_factor);
    } else {
      auto shift_excess = make_with_random_values<tnsr::I<DataVector, 3>>(
          make_not_null(&gen), make_not_null(&dist), num_points);
      auto deriv_shift_excess =
          make_with_random_values<tnsr::iJ<DataVector, 3>>(
              make_not_null(&gen), make_not_null(&dist), num_points);
      tnsr::II<DataVector, 3> longitudinal_shift_excess{
          num_points, std::numeric_limits<double>::signaling_NaN()};
      Xcts::longitudinal_operator_flat_cartesian(
          make_not_null(&longitudinal_shift_excess), deriv_shift_excess);
      tnsr::I<DataVector, 3> n_dot_longitudinal_shift_excess{
          num_points, std::numeric_limits<double>::signaling_NaN()};
      normal_dot_flux(make_not_null(&n_dot_longitudinal_shift_excess),
                      face_normal, longitudinal_shift_excess);
      elliptic::apply_boundary_condition<Linearized, void,
                                         tmpl::list<Robin<EnabledEquations>>>(
          boundary_condition, box, direction,
          make_not_null(&conformal_factor_minus_one),
          make_not_null(&lapse_times_conformal_factor_minus_one),
          make_not_null(&shift_excess),
          make_not_null(&n_dot_conformal_factor_gradient),
          make_not_null(&n_dot_lapse_times_conformal_factor_gradient),
          make_not_null(&n_dot_longitudinal_shift_excess),
          deriv_conformal_factor, deriv_lapse_times_conformal_factor,
          deriv_shift_excess);
      const auto expected_n_dot_longitudinal_shift_excess =
          pypp::call<tnsr::I<DataVector, 3>>(
              py_module, "robin_boundary_condition_shift", shift_excess,
              deriv_shift_excess, x, face_normal);
      CHECK_ITERABLE_APPROX(get<0>(n_dot_longitudinal_shift_excess),
                            get<0>(expected_n_dot_longitudinal_shift_excess));
      CHECK_ITERABLE_APPROX(get<1>(n_dot_longitudinal_shift_excess),
                            get<1>(expected_n_dot_longitudinal_shift_excess));
      CHECK_ITERABLE_APPROX(get<2>(n_dot_longitudinal_shift_excess),
                            get<2>(expected_n_dot_longitudinal_shift_excess));
    }
    const auto expected_n_dot_lapse_times_conformal_factor_gradient =
        pypp::call<Scalar<DataVector>>(
            py_module, "robin_boundary_condition_scalar",
            lapse_times_conformal_factor_minus_one, x);
    CHECK_ITERABLE_APPROX(
        get(n_dot_lapse_times_conformal_factor_gradient),
        get(expected_n_dot_lapse_times_conformal_factor_gradient));
  }
  const auto expected_n_dot_conformal_factor_gradient =
      pypp::call<Scalar<DataVector>>(py_module,
                                     "robin_boundary_condition_scalar",
                                     conformal_factor_minus_one, x);
  CHECK_ITERABLE_APPROX(get(n_dot_conformal_factor_gradient),
                        get(expected_n_dot_conformal_factor_gradient));
}

template <Xcts::Equations EnabledEquations>
void test_suite() {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>,
      Robin<EnabledEquations>>("Robin");
  REQUIRE(dynamic_cast<const Robin<EnabledEquations>*>(created.get()) !=
          nullptr);
  const auto& boundary_condition =
      dynamic_cast<const Robin<EnabledEquations>&>(*created);
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  {
    INFO("Properties");
    if constexpr (EnabledEquations == Xcts::Equations::Hamiltonian) {
      CHECK(boundary_condition.boundary_condition_types() ==
            std::vector<elliptic::BoundaryConditionType>{
                1, elliptic::BoundaryConditionType::Neumann});
    } else if constexpr (EnabledEquations ==
                         Xcts::Equations::HamiltonianAndLapse) {
      CHECK(boundary_condition.boundary_condition_types() ==
            std::vector<elliptic::BoundaryConditionType>{
                2, elliptic::BoundaryConditionType::Neumann});
    } else if constexpr (EnabledEquations ==
                         Xcts::Equations::HamiltonianLapseAndShift) {
      CHECK(boundary_condition.boundary_condition_types() ==
            std::vector<elliptic::BoundaryConditionType>{
                5, elliptic::BoundaryConditionType::Neumann});
    }
  }
  test_robin<EnabledEquations, false>(boundary_condition);
  test_robin<EnabledEquations, true>(boundary_condition);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Xcts.BoundaryConditions.Robin", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env("");
  test_suite<Xcts::Equations::Hamiltonian>();
  test_suite<Xcts::Equations::HamiltonianAndLapse>();
  test_suite<Xcts::Equations::HamiltonianLapseAndShift>();
}

}  // namespace Xcts::BoundaryConditions
