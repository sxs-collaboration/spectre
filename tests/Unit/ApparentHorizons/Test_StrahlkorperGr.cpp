// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperDataBox.hpp"  // IWYU pragma: keep
#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"   // IWYU prgma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/GrTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ApparentHorizons/StrahlkorperGrTestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
template <typename Solution, typename Fr, typename ExpectedLambda>
void test_expansion(const Solution& solution,
                    const Strahlkorper<Fr>& strahlkorper,
                    const ExpectedLambda& expected) noexcept {
  // Make databox from surface
  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);

  const double t = 0.0;
  const auto& cart_coords =
      db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);

  const auto vars = solution.variables(
      cart_coords, t, typename Solution::template tags<DataVector>{});

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(vars);
  const auto& deriv_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(vars);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const DataVector one_over_one_form_magnitude =
      1.0 / get(magnitude(
                db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                inverse_spatial_metric));
  const auto unit_normal_one_form = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
      one_over_one_form_magnitude);
  const auto grad_unit_normal_one_form =
      StrahlkorperGr::grad_unit_normal_one_form(
          db::get<StrahlkorperTags::Rhat<Frame::Inertial>>(box),
          db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box),
          unit_normal_one_form,
          db::get<StrahlkorperTags::D2xRadius<Frame::Inertial>>(box),
          one_over_one_form_magnitude,
          raise_or_lower_first_index(
              gr::christoffel_first_kind(deriv_spatial_metric),
              inverse_spatial_metric));
  const auto inverse_surface_metric = StrahlkorperGr::inverse_surface_metric(
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric),
      inverse_spatial_metric);

  const auto residual = StrahlkorperGr::expansion(
      grad_unit_normal_one_form, inverse_surface_metric,
      gr::extrinsic_curvature(
          get<gr::Tags::Lapse<3, Frame::Inertial, DataVector>>(vars),
          get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(vars),
          get<Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(vars),
          spatial_metric,
          get<Tags::dt<
              gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(vars),
          deriv_spatial_metric));

  Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get(residual), expected(get(residual).size()),
                               custom_approx);
}

namespace TestExtrinsicCurvature {
void test_minkowski() {
  // Make surface of radius 2.
  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(
      Strahlkorper<Frame::Inertial>(8, 8, 2.0, {{0.0, 0.0, 0.0}}));

  const double t = 0.0;
  const auto& cart_coords =
      db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);

  EinsteinSolutions::Minkowski<3> solution{};

  const auto deriv_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(
          solution.variables(
              cart_coords, t,
              tmpl::list<Tags::deriv<
                  gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>>{}));
  const auto inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(
          solution.variables(
              cart_coords, t,
              tmpl::list<gr::Tags::InverseSpatialMetric<3, Frame::Inertial,
                                                        DataVector>>{}));

  const DataVector one_over_one_form_magnitude =
      1.0 / get(magnitude(
                db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                inverse_spatial_metric));
  const auto unit_normal_one_form = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
      one_over_one_form_magnitude);
  const auto grad_unit_normal_one_form =
      StrahlkorperGr::grad_unit_normal_one_form(
          db::get<StrahlkorperTags::Rhat<Frame::Inertial>>(box),
          db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box),
          unit_normal_one_form,
          db::get<StrahlkorperTags::D2xRadius<Frame::Inertial>>(box),
          one_over_one_form_magnitude,
          raise_or_lower_first_index(
              gr::christoffel_first_kind(deriv_spatial_metric),
              inverse_spatial_metric));

  const auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);
  const auto extrinsic_curvature = StrahlkorperGr::extrinsic_curvature(
      grad_unit_normal_one_form, unit_normal_one_form, unit_normal_vector);
  const auto extrinsic_curvature_minkowski =
      TestHelpers::Minkowski::extrinsic_curvature_sphere(cart_coords);

  CHECK_ITERABLE_APPROX(extrinsic_curvature, extrinsic_curvature_minkowski);
}
}  // namespace TestExtrinsicCurvature

template <typename Solution, typename SpatialRicciScalar,
          typename ExpectedLambda>
void test_ricci_scalar(const Solution& solution,
                       const SpatialRicciScalar& spatial_ricci_scalar,
                       const ExpectedLambda& expected) noexcept {
  // Make surface of radius 2.
  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(
      Strahlkorper<Frame::Inertial>(8, 8, 2.0, {{0.0, 0.0, 0.0}}));

  const double t = 0.0;
  const auto& cart_coords =
      db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);

  const auto vars = solution.variables(
      cart_coords, t, typename Solution::template tags<DataVector>{});

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(vars);
  const auto& deriv_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(vars);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const DataVector one_over_one_form_magnitude =
      1.0 / get(magnitude(
                db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                inverse_spatial_metric));
  const auto unit_normal_one_form = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
      one_over_one_form_magnitude);
  const auto grad_unit_normal_one_form =
      StrahlkorperGr::grad_unit_normal_one_form(
          db::get<StrahlkorperTags::Rhat<Frame::Inertial>>(box),
          db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box),
          unit_normal_one_form,
          db::get<StrahlkorperTags::D2xRadius<Frame::Inertial>>(box),
          one_over_one_form_magnitude,
          raise_or_lower_first_index(
              gr::christoffel_first_kind(deriv_spatial_metric),
              inverse_spatial_metric));

  const auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);
  const auto ricci_scalar = StrahlkorperGr::ricci_scalar(
      spatial_ricci_scalar(cart_coords), unit_normal_vector,
      StrahlkorperGr::extrinsic_curvature(
          grad_unit_normal_one_form, unit_normal_one_form, unit_normal_vector),
      inverse_spatial_metric);

  CHECK_ITERABLE_APPROX(get(ricci_scalar), expected(get(ricci_scalar).size()));
}

template <typename Solution, typename ExpectedLambda>
void test_area_element(const Solution& solution, const double& surface_radius,
                       const ExpectedLambda& expected) noexcept {
  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(
      Strahlkorper<Frame::Inertial>(8, 8, surface_radius, {{0.0, 0.0, 0.0}}));

  const double t = 0.0;
  const auto& cart_coords =
      db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);

  const auto vars = solution.variables(
      cart_coords, t, typename Solution::template tags<DataVector>{});

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(vars);

  const auto& normal_one_form =
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box);
  const auto& r_hat = db::get<StrahlkorperTags::Rhat<Frame::Inertial>>(box);
  const auto& radius = db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
  const auto& jacobian =
      db::get<StrahlkorperTags::Jacobian<Frame::Inertial>>(box);

  const auto area_element = StrahlkorperGr::area_element(
      spatial_metric, jacobian, normal_one_form, radius, r_hat);

  CHECK_ITERABLE_APPROX(get(area_element), expected(get(area_element).size()));
}

template <typename Solution, typename Fr>
void test_area(const Solution& solution, const Strahlkorper<Fr>& strahlkorper,
               const double expected) noexcept {
  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);

  const double t = 0.0;
  const auto& cart_coords =
      db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);

  const auto vars = solution.variables(
      cart_coords, t, typename Solution::template tags<DataVector>{});

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(vars);

  const auto& normal_one_form =
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box);
  const auto& r_hat = db::get<StrahlkorperTags::Rhat<Frame::Inertial>>(box);
  const auto& radius = db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
  const auto& jacobian =
      db::get<StrahlkorperTags::Jacobian<Frame::Inertial>>(box);

  const auto area_element = StrahlkorperGr::area_element(
      spatial_metric, jacobian, normal_one_form, radius, r_hat);

  const double area =
      strahlkorper.ylm_spherepack().definite_integral(get(area_element).data());

  CHECK_ITERABLE_APPROX(area, expected);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperGr.Expansion",
                  "[ApparentHorizons][Unit]") {
  const auto sphere =
      Strahlkorper<Frame::Inertial>(8, 8, 2.0, {{0.0, 0.0, 0.0}});

  test_expansion(
      EinsteinSolutions::KerrSchild{1.0, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}},
      sphere, [](const size_t size) noexcept { return DataVector(size, 0.0); });
  test_expansion(
      EinsteinSolutions::Minkowski<3>{},
      sphere, [](const size_t size) noexcept { return DataVector(size, 1.0); });

  constexpr int l_max = 20;
  const double mass = 4.444;
  const std::array<double, 3> spin{{0.3, 0.4, 0.5}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  const auto horizon_radius = TestHelpers::Kerr::horizon_radius(
      Strahlkorper<Frame::Inertial>(l_max, l_max, 2.0, center)
          .ylm_spherepack()
          .theta_phi_points(),
      mass, spin);

  const auto kerr_horizon =
      Strahlkorper<Frame::Inertial>(l_max, l_max, get(horizon_radius), center);

  test_expansion(EinsteinSolutions::KerrSchild{mass, spin, center},
                 kerr_horizon, [](const size_t size) noexcept {
                   return DataVector(size, 0.0);
                 });
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperGr.ExtrinsicCurvature",
                  "[ApparentHorizons][Unit]") {
  // N.B.: test_minkowski() fully tests the extrinsic curvature function.
  // All components of extrinsic curvature of a sphere in flat space
  // are nontrivial; cf. extrinsic_curvature_sphere()
  // in StrahlkorperGrTestHelpers.cpp).
  TestExtrinsicCurvature::test_minkowski();
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperGr.RicciScalar",
                  "[ApparentHorizons][Unit]") {
  const double mass = 1.0;
  test_ricci_scalar(
      EinsteinSolutions::KerrSchild(mass, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}),
      [&mass](const auto& cartesian_coords) noexcept {
        return TestHelpers::Schwarzschild::spatial_ricci(cartesian_coords,
                                                         mass);
      },
      [&mass](const size_t size) noexcept {
        return DataVector(size, 0.5 / square(mass));
      });
  test_ricci_scalar(
      EinsteinSolutions::Minkowski<3>{},
      [](const auto& cartesian_coords) noexcept {
        return make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
            cartesian_coords, 0.0);
      },
      [](const size_t size) noexcept { return DataVector(size, 0.5); });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperGr.AreaElement",
                  "[ApparentHorizons][Unit]") {
  // Check value of dA for a Schwarzschild horizon and a sphere in flat space
  test_area_element(
      EinsteinSolutions::KerrSchild{4.0, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}},
      8.0, [](const size_t size) noexcept { return DataVector(size, 64.0); });
  test_area_element(
      EinsteinSolutions::Minkowski<3>{},
      2.0, [](const size_t size) noexcept { return DataVector(size, 4.0); });

  // Check the area of a Kerr horizon
  constexpr int l_max = 20;
  const double mass = 4.444;
  const std::array<double, 3> spin{{0.4, 0.33, 0.22}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  const double kerr_horizon_radius =
      mass * (1.0 + sqrt(1.0 - square(magnitude(spin))));
  // Eq. (26.84a) of Thorne and Blandford
  const double expected_area = 8.0 * M_PI * mass * kerr_horizon_radius;

  const auto horizon_radius = TestHelpers::Kerr::horizon_radius(
      Strahlkorper<Frame::Inertial>(l_max, l_max, 2.0, center)
          .ylm_spherepack()
          .theta_phi_points(),
      mass, spin);

  const auto kerr_horizon =
      Strahlkorper<Frame::Inertial>(l_max, l_max, get(horizon_radius), center);

  test_area(EinsteinSolutions::KerrSchild{mass, spin, center}, kerr_horizon,
            expected_area);
}
