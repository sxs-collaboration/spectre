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
#include "Elliptic/Systems/Xcts/BoundaryConditions/SuperposedBoostedBinary.hpp"
#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::BoundaryConditions {

namespace {

const std::string py_module{
    "Elliptic.Systems.Xcts.BoundaryConditions.SuperposedBoostedBinary"};

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

struct FactoryMetavars {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<elliptic::BoundaryConditions::BoundaryCondition<3>,
                             tmpl::list<SuperposedBoostedBinary<
                                 elliptic::analytic_data::AnalyticSolution,
                                 tmpl::list<Xcts::Solutions::Schwarzschild>>>>,
                  tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                             tmpl::list<Xcts::Solutions::Schwarzschild>>>;
  };
};

template <bool Linearized>
void test_suite(const std::string& options_string) {
  INFO("Test factory-creation");
  using IsolatedObjectBase = elliptic::analytic_data::AnalyticSolution;
  using IsolatedObjectClasses = tmpl::list<Xcts::Solutions::Schwarzschild>;
  register_classes_with_charm<Xcts::Solutions::Schwarzschild>();
  register_factory_classes_with_charm<FactoryMetavars>();
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<elliptic::BoundaryConditions::BoundaryCondition<3>>,
      FactoryMetavars>(options_string);
  const auto serialized = serialize_and_deserialize(created);
  REQUIRE(dynamic_cast<const SuperposedBoostedBinary<IsolatedObjectBase,
                                                     IsolatedObjectClasses>*>(
              serialized.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const SuperposedBoostedBinary<
      IsolatedObjectBase, IsolatedObjectClasses>&>(*serialized);
  {
    INFO("Properties");
    CHECK(boundary_condition.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              5, elliptic::BoundaryConditionType::Dirichlet});
  }
  {
    MAKE_GENERATOR(gen);
    std::uniform_real_distribution<> dist(-1., 1.);
    const size_t num_points = 3;
    const auto direction = Direction<3>::upper_zeta();
    const std::array<double, 3> center{{0., 0., 0.}};
    const auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
        make_not_null(&gen), make_not_null(&dist), num_points);
    const auto face_normal =
        make_spherical_face_normal_flat_cartesian(x, center);
    const auto box = db::create<db::AddSimpleTags<
        domain::Tags::Faces<3, domain::Tags::Coordinates<3, Frame::Inertial>>,
        domain::Tags::Faces<3, domain::Tags::FaceNormal<3, Frame::Inertial>>>>(
        DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, x}},
        DirectionMap<3, tnsr::i<DataVector, 3>>{{direction, face_normal}});
    auto conformal_factor_minus_one =
        make_with_random_values<Scalar<DataVector>>(
            make_not_null(&gen), make_not_null(&dist), num_points);
    auto lapse_times_conformal_factor_minus_one =
        make_with_random_values<Scalar<DataVector>>(
            make_not_null(&gen), make_not_null(&dist), num_points);
    auto shift_excess = make_with_random_values<tnsr::I<DataVector, 3>>(
        make_not_null(&gen), make_not_null(&dist), num_points);
    auto n_dot_conformal_factor_gradient =
        make_with_random_values<Scalar<DataVector>>(
            make_not_null(&gen), make_not_null(&dist), num_points);
    auto n_dot_lapse_times_conformal_factor_gradient =
        make_with_random_values<Scalar<DataVector>>(
            make_not_null(&gen), make_not_null(&dist), num_points);
    auto n_dot_longitudinal_shift_excess =
        make_with_random_values<tnsr::I<DataVector, 3>>(
            make_not_null(&gen), make_not_null(&dist), num_points);
    const tnsr::i<DataVector, 3> deriv_conformal_factor{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    const tnsr::i<DataVector, 3> deriv_lapse_times_conformal_factor{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    const tnsr::iJ<DataVector, 3> deriv_shift_excess{
        num_points, std::numeric_limits<double>::signaling_NaN()};

    elliptic::apply_boundary_condition<
        Linearized, void,
        tmpl::list<SuperposedBoostedBinary<IsolatedObjectBase,
                                           IsolatedObjectClasses>>>(
        boundary_condition, box, direction,
        make_not_null(&conformal_factor_minus_one),
        make_not_null(&lapse_times_conformal_factor_minus_one),
        make_not_null(&shift_excess),
        make_not_null(&n_dot_conformal_factor_gradient),
        make_not_null(&n_dot_lapse_times_conformal_factor_gradient),
        make_not_null(&n_dot_longitudinal_shift_excess), deriv_conformal_factor,
        deriv_lapse_times_conformal_factor, deriv_shift_excess);

    const auto expected_conformal_factor_minus_one =
        pypp::call<Scalar<DataVector>>(py_module, "conformal_factor_minus_one",
                                       x);
    const auto expected_lapse_times_conformal_factor_minus_one =
        pypp::call<Scalar<DataVector>>(
            py_module, "lapse_times_conformal_factor_minus_one", x);
    const auto expected_shift_excess =
        pypp::call<tnsr::I<DataVector, 3>>(py_module, "shift_excess", x);

    CHECK_ITERABLE_APPROX(get(conformal_factor_minus_one),
                          get(expected_conformal_factor_minus_one));
    CHECK_ITERABLE_APPROX(get(lapse_times_conformal_factor_minus_one),
                          get(expected_lapse_times_conformal_factor_minus_one));
    CHECK_ITERABLE_APPROX(get<0>(shift_excess), get<0>(expected_shift_excess));
    CHECK_ITERABLE_APPROX(get<1>(shift_excess), get<1>(expected_shift_excess));
    CHECK_ITERABLE_APPROX(get<2>(shift_excess), get<2>(expected_shift_excess));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Xcts.BoundaryConditions.SuperposedBoostedBinary",
                  "[Unit][Elliptic]") {
  const pypp::SetupLocalPythonEnvironment local_python_env("");
  test_suite<false>(
      "SuperposedBoostedBinary:\n"
      "  XCoords: [-5., 6.]\n"
      "  Masses: [1.1, 0.43]\n"
      "  MomentumRight: [-0.01, -0.01, -0.01]\n"
      "  CenterOfMassOffset: [0.02, 0.01]\n"
      "  ObjectLeft:\n"
      "    Schwarzschild:\n"
      "      Mass: 1.1\n"
      "      Coordinates: Isotropic\n"
      "  ObjectRight:\n"
      "    Schwarzschild:\n"
      "      Mass: 0.43\n"
      "      Coordinates: Isotropic");
}

}  // namespace Xcts::BoundaryConditions
