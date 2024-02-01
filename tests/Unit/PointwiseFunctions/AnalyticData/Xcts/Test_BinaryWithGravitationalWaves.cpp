// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <optional>
#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/Xcts/VerifySolution.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::AnalyticData {
namespace {

using test_tags = tmpl::list<
    detail::Tags::DistanceLeft<DataVector>,
    detail::Tags::DistanceRight<DataVector>,
    ::Tags::deriv<detail::Tags::OneOverDistanceLeft<DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>,
    ::Tags::deriv<detail::Tags::OneOverDistanceRight<DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>,
    ::Tags::deriv<
        ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceLeft<DataVector>,
                                    tmpl::size_t<3>, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>,
    ::Tags::deriv<
        ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceRight<DataVector>,
                                    tmpl::size_t<3>, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>,
    detail::Tags::NormalLeft<DataVector>, detail::Tags::NormalRight<DataVector>,
    detail::Tags::RadiativeTerm<DataVector>,
    detail::Tags::NearZoneTerm<DataVector>,
    detail::Tags::PresentTerm<DataVector>,
    detail::Tags::PostNewtonianConjugateMomentum3<DataVector>,
    detail::Tags::PostNewtonianExtrinsicCurvature<DataVector>,
    Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                            Frame::Inertial>,
    Tags::ConformalFactorMinusOne<DataVector>,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataVector>, 0>,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataVector, 3>, 0>>;

struct BinaryWithGravitationalWavesProxy {
  tuples::tagged_tuple_from_typelist<test_tags> test_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
    return binary.variables(x, test_tags{});
  }

  const BinaryWithGravitationalWaves& binary;
};

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<elliptic::analytic_data::Background,
                             tmpl::list<BinaryWithGravitationalWaves>>>;
  };
};

void test_data(const double mass_left, const double mass_right,
               const double xcoord_left, const double xcoord_right,
               const double attenuation_parameter, const double outer_radius,
               const bool write_evolution_option,
               const std::string& options_string) {
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<elliptic::analytic_data::Background>, Metavariables>(
      options_string);
  REQUIRE(dynamic_cast<const BinaryWithGravitationalWaves*>(created.get()) !=
          nullptr);
  const auto& derived =
      dynamic_cast<const BinaryWithGravitationalWaves&>(*created);
  const auto binary = serialize_and_deserialize(derived);
  {
    INFO("Properties");
    CHECK(binary.mass_left() == mass_left);
    CHECK(binary.mass_right() == mass_right);
    CHECK(binary.xcoord_left() == xcoord_left);
    CHECK(binary.xcoord_right() == xcoord_right);
    CHECK(binary.attenuation_parameter() == attenuation_parameter);
    CHECK(binary.outer_radius() == outer_radius);
    CHECK(binary.write_evolution_option() == write_evolution_option);
  }
  {
    INFO("Check derivative");
    using Affine = domain::CoordinateMaps::Affine;
    using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine,
  Affine>;
    // Setup grid
    const size_t num_points_1d = 5;
    const std::array<double, 3> lower_bound{{1.999, 0.001, 0.001}};
    const std::array<double, 3> upper_bound{{2.001, -0.001, -0.001}};
    Mesh<3> mesh{num_points_1d, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::ElementLogical,
  Frame::Inertial>(Affine3D{ Affine{-1., 1., lower_bound[0], upper_bound[0]},
            Affine{-1., 1., lower_bound[1], upper_bound[1]},
            Affine{-1., 1., lower_bound[2], upper_bound[2]},
        });
    const size_t num_points_3d = num_points_1d * num_points_1d * num_points_1d;
    const DataVector used_for_size =
        DataVector(num_points_3d, std::numeric_limits<double>::signaling_NaN());
    // Setup coordinates
    const auto x_logical = logical_coordinates(mesh);
    const auto x = coord_map(x_logical);
    const auto inv_jacobian = coord_map.inv_jacobian(x_logical);
    // Check the derivatives
    tnsr::I<DataVector, 3> v_right(x);
    v_right.get(0) -= xcoord_right;
    tnsr::I<DataVector, 3> v_left(x);
    v_left.get(0) -= xcoord_left;
    const auto distance_right = magnitude(v_right);
    const auto distance_left = magnitude(v_left);
    const auto one_over_distance_right_aux = 1. / get(distance_right);
    const auto one_over_distance_left_aux = 1. / get(distance_left);
    const Scalar<DataVector> one_over_distance_right{
        one_over_distance_right_aux};
    const Scalar<DataVector> one_over_distance_left{one_over_distance_left_aux};
    const auto deriv_1_distance_right =
        partial_derivative(distance_right, mesh, inv_jacobian);
    const auto deriv_1_distance_left =
        partial_derivative(distance_left, mesh, inv_jacobian);
    const auto deriv_2_distance_right =
        partial_derivative(deriv_1_distance_right, mesh, inv_jacobian);
    const auto deriv_2_distance_left =
        partial_derivative(deriv_1_distance_left, mesh, inv_jacobian);
    const auto deriv_3_distance_right_test =
        partial_derivative(deriv_2_distance_right, mesh, inv_jacobian);
    const auto deriv_3_distance_left_test =
        partial_derivative(deriv_2_distance_left, mesh, inv_jacobian);
    const auto deriv_one_over_distance_right_test =
        partial_derivative(one_over_distance_right, mesh, inv_jacobian);
    const auto deriv_one_over_distance_left_test =
        partial_derivative(one_over_distance_left, mesh, inv_jacobian);
    const auto deriv_variables = binary.variables(
        x, mesh, inv_jacobian,
        tmpl::list<
            ::Tags::deriv<detail::Tags::OneOverDistanceLeft<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>,
            ::Tags::deriv<detail::Tags::OneOverDistanceRight<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>,
            ::Tags::deriv<
                ::Tags::deriv<
                    ::Tags::deriv<detail::Tags::DistanceRight<DataVector>,
                                  tmpl::size_t<3>, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
                tmpl::size_t<3>, Frame::Inertial>,
            ::Tags::deriv<
                ::Tags::deriv<
                    ::Tags::deriv<detail::Tags::DistanceLeft<DataVector>,
                                  tmpl::size_t<3>, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
                tmpl::size_t<3>, Frame::Inertial>>{});
    const auto deriv_3_distance_right = get<::Tags::deriv<
        ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceRight<DataVector>,
                                    tmpl::size_t<3>, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>>(deriv_variables);
    const auto deriv_3_distance_left = get<::Tags::deriv<
        ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceLeft<DataVector>,
                                    tmpl::size_t<3>, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>>(deriv_variables);
    const auto deriv_one_over_distance_right =
        get<::Tags::deriv<detail::Tags::OneOverDistanceRight<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(deriv_variables);
    const auto deriv_one_over_distance_left =
        get<::Tags::deriv<detail::Tags::OneOverDistanceLeft<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(deriv_variables);

    Approx approx = Approx::custom().epsilon(1e-3).scale(1.);
    CHECK_ITERABLE_CUSTOM_APPROX(deriv_3_distance_right_test,
                                 deriv_3_distance_right, approx);
    CHECK_ITERABLE_CUSTOM_APPROX(deriv_3_distance_left_test,
                                 deriv_3_distance_left, approx);
    CHECK_ITERABLE_CUSTOM_APPROX(deriv_one_over_distance_right_test,
                                 deriv_one_over_distance_right, approx);
    CHECK_ITERABLE_CUSTOM_APPROX(deriv_one_over_distance_left_test,
                                 deriv_one_over_distance_left, approx);
  }
  {
    const BinaryWithGravitationalWavesProxy proxy{binary};
    pypp::check_with_random_values<1>(
        &BinaryWithGravitationalWavesProxy::test_variables, proxy,
        "BinaryWithGravitationalWaves",
        {"distance_left",
         "distance_right",
         "deriv_one_over_distance_left",
         "deriv_one_over_distance_right",
         "deriv_3_distance_left",
         "deriv_3_distance_right",
         "normal_left",
         "normal_right",
         "radiative_term",
         "near_zone_term",
         "present_term",
         "pn_conjugate_momentum3",
         "pn_extrinsic_curvature",
         "conformal_metric",
         "extrinsic_curvature_trace",
         "shift_background",
         "longitudinal_shift_background",
         "conformal_factor_minus_one",
         "energy_density",
         "stress_trace",
         "momentum_density"},
        {{{-10. + xcoord_left, xcoord_right + 10.}}}, std::make_tuple(),
        DataVector(5));
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.Xcts.BinaryWithGravitationalWaves",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/Xcts"};
  test_data(1.1, .9, -4.5, 10.2, .99, 21., false,
            "BinaryWithGravitationalWaves:\n"
            "  MassLeft: 1.1\n"
            "  MassRight: .9\n"
            "  XCoordsLeft: -4.5\n"
            "  XCoordsRight: 10.2\n"
            "  AttenuationParameter: .99\n"
            "  OuterRadius: 21.\n"
            "  WriteEvolutionOption: False");
}

}  // namespace Xcts::AnalyticData
