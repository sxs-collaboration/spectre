// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {
namespace {

using field_tags =
    tmpl::list<Tags::ConformalFactor<DataVector>,
               Tags::LapseTimesConformalFactor<DataVector>,
               Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>;
using auxiliary_field_tags =
    tmpl::list<::Tags::deriv<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                             Frame::Inertial>,
               ::Tags::deriv<Tags::LapseTimesConformalFactor<DataVector>,
                             tmpl::size_t<3>, Frame::Inertial>,
               Tags::ShiftStrain<DataVector, 3, Frame::Inertial>>;
using flux_tags =
    tmpl::list<::Tags::Flux<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                            Frame::Inertial>,
               ::Tags::Flux<Tags::LapseTimesConformalFactor<DataVector>,
                            tmpl::size_t<3>, Frame::Inertial>,
               Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>>;
using background_tags = tmpl::list<
    Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
    Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
    ::Tags::deriv<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>,
    Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                            Frame::Inertial>>;
using matter_source_tags = tmpl::list<
    Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>,
    Tags::Conformal<gr::Tags::StressTrace<DataVector>, 0>,
    Tags::Conformal<gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>,
                    0>>;
using gr_tags = tmpl::list<gr::Tags::Lapse<DataVector>,
                           gr::Tags::Shift<3, Frame::Inertial, DataVector>>;
using derived_tags = tmpl::list<
    Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>,
    Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataVector>,
    Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<DataVector>>;

struct SchwarzschildProxy : Xcts::Solutions::Schwarzschild<> {
  using Xcts::Solutions::Schwarzschild<>::Schwarzschild;
  tuples::tagged_tuple_from_typelist<
      tmpl::append<field_tags, auxiliary_field_tags>>
  field_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(
        x, tmpl::append<field_tags, auxiliary_field_tags>{});
  }
  tuples::tagged_tuple_from_typelist<flux_tags> flux_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(x, flux_tags{});
  }
  tuples::tagged_tuple_from_typelist<background_tags> background_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(x, background_tags{});
  }
  tuples::tagged_tuple_from_typelist<matter_source_tags>
  matter_source_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(x, matter_source_tags{});
  }
  tuples::tagged_tuple_from_typelist<gr_tags> gr_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(x, gr_tags{});
  }
  tuples::tagged_tuple_from_typelist<derived_tags> derived_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
    return Xcts::Solutions::Schwarzschild<>::variables(x, derived_tags{});
  }
};

void test_solution(const double mass,
                   const Xcts::Solutions::SchwarzschildCoordinates coords,
                   const double expected_radius_at_horizon,
                   const std::string& py_functions_suffix,
                   const std::string& options_string) {
  CAPTURE(mass);
  CAPTURE(coords);
  const auto created =
      TestHelpers::test_factory_creation<Xcts::Solutions::AnalyticSolution<
          tmpl::list<Xcts::Solutions::Registrars::Schwarzschild>>>(
          options_string);
  REQUIRE(dynamic_cast<const Schwarzschild<>*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const Schwarzschild<>&>(*created);
  {
    INFO("Properties");
    CHECK(solution.mass() == mass);
    CHECK(solution.radius_at_horizon() == approx(expected_radius_at_horizon));
  }
  {
    INFO("Semantics");
    test_serialization(solution);
    test_copy_semantics(solution);
    auto move_solution = solution;
    test_move_semantics(std::move(move_solution), solution);
  }
  {
    INFO("Random-value tests");
    const SchwarzschildProxy solution_proxy{mass, coords};
    const double inner_radius = expected_radius_at_horizon;
    const double outer_radius = 3. * expected_radius_at_horizon;
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::field_variables, solution_proxy, "Schwarzschild",
        {"conformal_factor_" + py_functions_suffix,
         "lapse_times_conformal_factor_" + py_functions_suffix,
         "shift_" + py_functions_suffix,
         "conformal_factor_gradient_" + py_functions_suffix,
         "lapse_times_conformal_factor_gradient_" + py_functions_suffix,
         "shift_strain_" + py_functions_suffix},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::flux_variables, solution_proxy, "Schwarzschild",
        {"conformal_factor_gradient_" + py_functions_suffix,
         "lapse_times_conformal_factor_gradient_" + py_functions_suffix,
         "longitudinal_shift_" + py_functions_suffix},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::background_variables, solution_proxy,
        "Schwarzschild",
        {"conformal_metric_" + py_functions_suffix,
         "inv_conformal_metric_" + py_functions_suffix,
         "deriv_conformal_metric_" + py_functions_suffix,
         "extrinsic_curvature_trace_" + py_functions_suffix,
         "extrinsic_curvature_trace_gradient_" + py_functions_suffix,
         "shift_background",
         "longitudinal_shift_background_minus_dt_conformal_metric"},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::matter_source_variables, solution_proxy,
        "Schwarzschild", {"energy_density", "stress_trace", "momentum_density"},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::gr_variables, solution_proxy, "Schwarzschild",
        {"lapse_" + py_functions_suffix, "shift_" + py_functions_suffix},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::derived_variables, solution_proxy, "Schwarzschild",
        {"shift_dot_extrinsic_curvature_trace_gradient_" + py_functions_suffix,
         "longitudinal_shift_minus_dt_conformal_metric_square_" +
             py_functions_suffix,
         "longitudinal_shift_minus_dt_conformal_metric_over_lapse_square_" +
             py_functions_suffix},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
  }
  {
    INFO("Verify the solution solves the XCTS equations");
    // Once the XCTS equations are implemented we will check here that the
    // solution numerically solves the equations. That's both a rigorous test
    // that the solution is correctly implemented, as well as a test of the XCTS
    // system implementation. Until then we just call into the solution and
    // probe a few variables.
    const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    const double inner_radius = expected_radius_at_horizon;
    const double outer_radius = 3. * expected_radius_at_horizon;
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>
        coord_map{{{-1., 1., inner_radius, outer_radius},
                   {-1., 1., inner_radius, outer_radius},
                   {-1., 1., inner_radius, outer_radius}}};
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = coord_map(logical_coords);
    const auto inv_jacobian = coord_map.inv_jacobian(logical_coords);
    const auto vars = solution.variables(
        inertial_coords, mesh, inv_jacobian,
        tmpl::list<Tags::ConformalRicciScalar<DataVector>>{});
    CHECK_ITERABLE_APPROX(
        get(get<Tags::ConformalRicciScalar<DataVector>>(vars)),
        DataVector(num_points, 0.));
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Schwarzschild",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Xcts"};
  test_solution(1., SchwarzschildCoordinates::Isotropic, 0.5, "isotropic",
                "Schwarzschild:\n"
                "  Mass: 1.\n"
                "  Coordinates: Isotropic");
  test_solution(0.8, SchwarzschildCoordinates::Isotropic, 0.4, "isotropic",
                "Schwarzschild:\n"
                "  Mass: 0.8\n"
                "  Coordinates: Isotropic");
}

}  // namespace Xcts::Solutions
