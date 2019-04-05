// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/Tags.hpp"  // IWYU pragma: keep
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"   // IWYU prgma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ApparentHorizons/StrahlkorperGrTestHelpers.hpp"

// IWYU pragma: no_forward_declare Tags::deriv
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
          get<gr::Tags::Lapse<DataVector>>(vars),
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

  gr::Solutions::Minkowski<3> solution{};

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

void test_euclidean_area_element(
    const Strahlkorper<Frame::Inertial>& strahlkorper) noexcept {
  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);

  // Create a Minkowski metric.
  const gr::Solutions::Minkowski<3> solution{};
  const double t = 0.0;
  const auto& cart_coords =
      db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);
  const auto vars = solution.variables(
      cart_coords, t, gr::Solutions::Minkowski<3>::tags<DataVector>{});
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(vars);

  const auto& normal_one_form =
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box);
  const auto& r_hat = db::get<StrahlkorperTags::Rhat<Frame::Inertial>>(box);
  const auto& radius = db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
  const auto& jacobian =
      db::get<StrahlkorperTags::Jacobian<Frame::Inertial>>(box);

  // We are using a flat metric, so area_element and euclidean_area_element
  // should be the same, and this is what we test.
  const auto area_element = StrahlkorperGr::area_element(
      spatial_metric, jacobian, normal_one_form, radius, r_hat);
  const auto euclidean_area_element = StrahlkorperGr::euclidean_area_element(
      jacobian, normal_one_form, radius, r_hat);

  CHECK_ITERABLE_APPROX(get(euclidean_area_element), get(area_element));
}

template <typename Solution, typename Fr>
void test_area(const Solution& solution, const Strahlkorper<Fr>& strahlkorper,
               const double expected, const double expected_irreducible_mass,
               const double dimensionful_spin_magnitude,
               const double expected_christodoulou_mass) noexcept {
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

  const double irreducible_mass = StrahlkorperGr::irreducible_mass(area);
  CHECK(irreducible_mass == approx(expected_irreducible_mass));

  const double christodoulou_mass = StrahlkorperGr::christodoulou_mass(
      dimensionful_spin_magnitude, irreducible_mass);
  CHECK(christodoulou_mass == approx(expected_christodoulou_mass));
}

template <typename Solution, typename Fr>
void test_surface_integral_of_scalar(const Solution& solution,
                                     const Strahlkorper<Fr>& strahlkorper,
                                     const double expected) noexcept {
  const auto box =
      db::create<db::AddSimpleTags<StrahlkorperTags::items_tags<Fr>>,
                 db::AddComputeTags<StrahlkorperTags::compute_items_tags<Fr>>>(
          strahlkorper);

  const double t = 0.0;
  const auto& cart_coords = db::get<StrahlkorperTags::CartesianCoords<Fr>>(box);

  const auto vars = solution.variables(
      cart_coords, t, typename Solution::template tags<DataVector>{});

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Fr, DataVector>>(vars);

  const auto& normal_one_form =
      db::get<StrahlkorperTags::NormalOneForm<Fr>>(box);
  const auto& r_hat = db::get<StrahlkorperTags::Rhat<Fr>>(box);
  const auto& radius = db::get<StrahlkorperTags::Radius<Fr>>(box);
  const auto& jacobian = db::get<StrahlkorperTags::Jacobian<Fr>>(box);

  const auto area_element = StrahlkorperGr::area_element(
      spatial_metric, jacobian, normal_one_form, radius, r_hat);

  auto scalar = make_with_value<Scalar<DataVector>>(radius, 0.0);
  get(scalar) = square(get<0>(cart_coords));

  const double integral = StrahlkorperGr::surface_integral_of_scalar(
      area_element, scalar, strahlkorper);

  CHECK_ITERABLE_APPROX(integral, expected);
}

template <typename Solution, typename Fr>
void test_spin_function(const Solution& solution,
                        const Strahlkorper<Fr>& strahlkorper,
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
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const auto& normal_one_form =
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box);
  const DataVector one_over_one_form_magnitude =
      1.0 / get(magnitude(
                db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                inverse_spatial_metric));
  const auto unit_normal_one_form = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
      one_over_one_form_magnitude);
  const auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);

  const auto& r_hat = db::get<StrahlkorperTags::Rhat<Frame::Inertial>>(box);
  const auto& radius = db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
  const auto& jacobian =
      db::get<StrahlkorperTags::Jacobian<Frame::Inertial>>(box);
  const auto area_element = StrahlkorperGr::area_element(
      spatial_metric, jacobian, normal_one_form, radius, r_hat);

  const auto& ylm = strahlkorper.ylm_spherepack();

  const auto& deriv_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(vars);
  const auto& extrinsic_curvature = gr::extrinsic_curvature(
      get<gr::Tags::Lapse<DataVector>>(vars),
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(vars),
      get<Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(vars),
      spatial_metric,
      get<Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
          vars),
      deriv_spatial_metric);

  const auto& tangents =
      db::get<StrahlkorperTags::Tangents<Frame::Inertial>>(box);

  const auto& spin_function = StrahlkorperGr::spin_function(
      tangents, ylm, unit_normal_vector, area_element, extrinsic_curvature);

  auto integrand = spin_function;
  get(integrand) *= get(area_element) * get(spin_function);

  const double integral = ylm.definite_integral(get(integrand).data());

  Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(integral, expected, custom_approx);
}

template <typename Solution, typename Fr>
void test_dimensionful_spin_magnitude(
    const Solution& solution, const Strahlkorper<Fr>& strahlkorper,
    const double mass, const std::array<double, 3> dimensionless_spin,
    const Scalar<DataVector>& horizon_radius_with_spin_on_z_axis,
    const YlmSpherepack& ylm_with_spin_on_z_axis, const double expected,
    const double tolerance) noexcept {
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
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const auto& normal_one_form =
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box);
  const DataVector one_over_one_form_magnitude =
      1.0 / get(magnitude(
                db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                inverse_spatial_metric));
  const auto unit_normal_one_form = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
      one_over_one_form_magnitude);
  const auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);

  const auto& r_hat = db::get<StrahlkorperTags::Rhat<Frame::Inertial>>(box);
  const auto& radius = db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
  const auto& jacobian =
      db::get<StrahlkorperTags::Jacobian<Frame::Inertial>>(box);
  const auto area_element = StrahlkorperGr::area_element(
      spatial_metric, jacobian, normal_one_form, radius, r_hat);

  const auto& ylm = strahlkorper.ylm_spherepack();

  const auto& deriv_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(vars);
  const auto& extrinsic_curvature = gr::extrinsic_curvature(
      get<gr::Tags::Lapse<DataVector>>(vars),
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(vars),
      get<Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(vars),
      spatial_metric,
      get<Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
          vars),
      deriv_spatial_metric);

  const auto& tangents =
      db::get<StrahlkorperTags::Tangents<Frame::Inertial>>(box);

  const auto spin_function = StrahlkorperGr::spin_function(
      tangents, ylm, unit_normal_vector, area_element, extrinsic_curvature);

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

  const auto ricci_scalar = TestHelpers::Kerr::horizon_ricci_scalar(
      horizon_radius_with_spin_on_z_axis, ylm_with_spin_on_z_axis, ylm, mass,
      dimensionless_spin);

  const double spin_magnitude = StrahlkorperGr::dimensionful_spin_magnitude(
      ricci_scalar, spin_function, spatial_metric, tangents, ylm, area_element);

  Approx custom_approx = Approx::custom().epsilon(tolerance).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(spin_magnitude, expected, custom_approx);
}

template <typename Solution, typename Fr>
void test_spin_vector(
    const Solution& solution, const Strahlkorper<Fr>& strahlkorper,
    const double mass, const std::array<double, 3> dimensionless_spin,
    const Scalar<DataVector>& horizon_radius_with_spin_on_z_axis,
    const YlmSpherepack& ylm_with_spin_on_z_axis) noexcept {
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
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const auto& normal_one_form =
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box);
  const DataVector one_over_one_form_magnitude =
      1.0 / get(magnitude(
                db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                inverse_spatial_metric));
  const auto unit_normal_one_form = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
      one_over_one_form_magnitude);
  const auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);

  const auto& r_hat = db::get<StrahlkorperTags::Rhat<Frame::Inertial>>(box);
  const auto& radius = db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
  const auto& jacobian =
      db::get<StrahlkorperTags::Jacobian<Frame::Inertial>>(box);
  const auto area_element = StrahlkorperGr::area_element(
      spatial_metric, jacobian, normal_one_form, radius, r_hat);

  const auto& ylm = strahlkorper.ylm_spherepack();

  const auto& deriv_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(vars);
  const auto& extrinsic_curvature = gr::extrinsic_curvature(
      get<gr::Tags::Lapse<DataVector>>(vars),
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(vars),
      get<Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(vars),
      spatial_metric,
      get<Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
          vars),
      deriv_spatial_metric);

  const auto& tangents =
      db::get<StrahlkorperTags::Tangents<Frame::Inertial>>(box);

  const auto& spin_function = StrahlkorperGr::spin_function(
      tangents, ylm, unit_normal_vector, area_element, extrinsic_curvature);

  const auto ricci_scalar = TestHelpers::Kerr::horizon_ricci_scalar(
      horizon_radius_with_spin_on_z_axis, ylm_with_spin_on_z_axis, ylm, mass,
      dimensionless_spin);

  const auto spin_vector =
      StrahlkorperGr::spin_vector(magnitude(dimensionless_spin), area_element,
                                  Scalar<DataVector>{std::move(radius)}, r_hat,
                                  ricci_scalar, spin_function, ylm);

  CHECK_ITERABLE_APPROX(spin_vector, dimensionless_spin);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperGr.Expansion",
                  "[ApparentHorizons][Unit]") {
  const auto sphere =
      Strahlkorper<Frame::Inertial>(8, 8, 2.0, {{0.0, 0.0, 0.0}});

  test_expansion(
      gr::Solutions::KerrSchild{1.0, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}},
      sphere, [](const size_t size) noexcept { return DataVector(size, 0.0); });
  test_expansion(
      gr::Solutions::Minkowski<3>{},
      sphere, [](const size_t size) noexcept { return DataVector(size, 1.0); });

  constexpr int l_max = 20;
  const double mass = 4.444;
  const std::array<double, 3> spin{{0.3, 0.4, 0.5}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  const auto horizon_radius = gr::Solutions::kerr_horizon_radius(
      Strahlkorper<Frame::Inertial>(l_max, l_max, 2.0, center)
          .ylm_spherepack()
          .theta_phi_points(),
      mass, spin);

  const auto kerr_horizon =
      Strahlkorper<Frame::Inertial>(l_max, l_max, get(horizon_radius), center);

  test_expansion(gr::Solutions::KerrSchild{mass, spin, center},
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
      gr::Solutions::KerrSchild(mass, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}),
      [&mass](const auto& cartesian_coords) noexcept {
        return TestHelpers::Schwarzschild::spatial_ricci(cartesian_coords,
                                                         mass);
      },
      [&mass](const size_t size) noexcept {
        return DataVector(size, 0.5 / square(mass));
      });
  test_ricci_scalar(
      gr::Solutions::Minkowski<3>{},
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
      gr::Solutions::KerrSchild{4.0, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}},
      8.0, [](const size_t size) noexcept { return DataVector(size, 64.0); });
  test_area_element(
      gr::Solutions::Minkowski<3>{},
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
  const double expected_irreducible_mass =
      sqrt(0.5 * mass * kerr_horizon_radius);

  const auto horizon_radius = gr::Solutions::kerr_horizon_radius(
      Strahlkorper<Frame::Inertial>(l_max, l_max, 2.0, center)
          .ylm_spherepack()
          .theta_phi_points(),
      mass, spin);

  const auto kerr_horizon =
      Strahlkorper<Frame::Inertial>(l_max, l_max, get(horizon_radius), center);

  test_area(gr::Solutions::KerrSchild{mass, spin, center}, kerr_horizon,
            expected_area, expected_irreducible_mass,
            square(mass) * magnitude(spin), mass);

  test_euclidean_area_element(kerr_horizon);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperGr.SurfaceInteralOfScalar",
                  "[ApparentHorizons][Unit]") {
  // Check the surface integral of a Schwarzschild horizon, using the radius
  // as the scalar
  constexpr int l_max = 20;
  const double mass = 4.444;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  const double radius = 2.0 * mass;

  // The test will integrate x^2 on a Schwarzschild horizon.
  // This is the analytic result.
  const double expected_integral = 4.0 * M_PI * square(square(radius)) / 3.0;

  const auto horizon =
      Strahlkorper<Frame::Inertial>(l_max, l_max, radius, center);

  test_surface_integral_of_scalar(gr::Solutions::KerrSchild{mass, spin, center},
                                  horizon, expected_integral);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperGr.SpinFunction",
                  "[ApparentHorizons][Unit]") {
  // Set up Kerr horizon
  constexpr int l_max = 24;
  const double mass = 4.444;
  const std::array<double, 3> spin{{0.4, 0.33, 0.22}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  const auto horizon_radius = gr::Solutions::kerr_horizon_radius(
      Strahlkorper<Frame::Inertial>(l_max, l_max, 2.0, center)
          .ylm_spherepack()
          .theta_phi_points(),
      mass, spin);

  const auto kerr_horizon =
      Strahlkorper<Frame::Inertial>(l_max, l_max, get(horizon_radius), center);

  // Check value of SpinFunction^2 integrated over the surface for
  // Schwarzschild. Expected result is zero.
  test_spin_function(gr::Solutions::KerrSchild{mass, {{0.0, 0.0, 0.0}}, center},
                     Strahlkorper<Frame::Inertial>(l_max, l_max, 8.888, center),
                     0.0);

  // Check value of SpinFunction^2 integrated over the surface for
  // Kerr. Derive this by integrating the square of the imaginary
  // part of Newman-Penrose Psi^2 on the Kerr horizon using the Kinnersley
  // null tetrad. For example, this integral is set up in the section
  // ``Spin function for Kerr'' of Geoffrey Lovelace's overleaf note:
  // https://v2.overleaf.com/read/twdtxchyrtyv
  // The integral does not have a simple, closed form, so just
  // enter the expected numerical value for this test.
  const double expected_spin_function_sq_integral = 4.0 * 0.0125109627941394;

  test_spin_function(gr::Solutions::KerrSchild{mass, spin, center},
                     kerr_horizon, expected_spin_function_sq_integral);
}

SPECTRE_TEST_CASE(
    "Unit.ApparentHorizons.StrahlkorperGr.DimensionfulSpinMagnitude",
    "[ApparentHorizons][Unit]") {
  const double mass = 2.0;
  const std::array<double, 3> generic_dimensionless_spin = {{0.12, 0.08, 0.04}};
  const double expected_dimensionless_spin_magnitude =
      magnitude(generic_dimensionless_spin);
  const double expected_spin_magnitude =
      expected_dimensionless_spin_magnitude * square(mass);
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  const double aligned_tolerance = 1.e-13;
  const double aligned_l_max = 12;
  const std::array<double, 3> aligned_dimensionless_spin = {
      {0.0, 0.0, expected_dimensionless_spin_magnitude}};
  const auto aligned_horizon_radius = gr::Solutions::kerr_horizon_radius(
      Strahlkorper<Frame::Inertial>(aligned_l_max, aligned_l_max, 2.0, center)
          .ylm_spherepack()
          .theta_phi_points(),
      mass, aligned_dimensionless_spin);
  const auto aligned_kerr_horizon = Strahlkorper<Frame::Inertial>(
      aligned_l_max, aligned_l_max, get(aligned_horizon_radius), center);
  test_dimensionful_spin_magnitude(
      gr::Solutions::KerrSchild{mass, aligned_dimensionless_spin, center},
      aligned_kerr_horizon, mass, aligned_dimensionless_spin,
      aligned_horizon_radius, aligned_kerr_horizon.ylm_spherepack(),
      expected_spin_magnitude, aligned_tolerance);

  const double generic_tolerance = 1.e-13;
  const int generic_l_max = 12;
  const double expected_generic_dimensionless_spin_magnitude =
      magnitude(generic_dimensionless_spin);
  const double expected_generic_spin_magnitude =
      expected_generic_dimensionless_spin_magnitude * square(mass);
  const auto generic_horizon_radius = gr::Solutions::kerr_horizon_radius(
      Strahlkorper<Frame::Inertial>(generic_l_max, generic_l_max, 2.0, center)
          .ylm_spherepack()
          .theta_phi_points(),
      mass, generic_dimensionless_spin);
  const auto generic_kerr_horizon = Strahlkorper<Frame::Inertial>(
      generic_l_max, generic_l_max, get(generic_horizon_radius), center);

  // Create rotated horizon radius, Strahlkorper, with same spin magnitude
  // but with spin on the z axis
  const auto rotated_horizon_radius = gr::Solutions::kerr_horizon_radius(
      Strahlkorper<Frame::Inertial>(generic_l_max, generic_l_max, 2.0, center)
          .ylm_spherepack()
          .theta_phi_points(),
      mass, aligned_dimensionless_spin);
  const auto rotated_kerr_horizon = Strahlkorper<Frame::Inertial>(
      generic_l_max, generic_l_max, get(rotated_horizon_radius), center);
  test_dimensionful_spin_magnitude(
      gr::Solutions::KerrSchild{mass, generic_dimensionless_spin, center},
      generic_kerr_horizon, mass, generic_dimensionless_spin,
      rotated_horizon_radius, rotated_kerr_horizon.ylm_spherepack(),
      expected_generic_spin_magnitude, generic_tolerance);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperGr.SpinVector",
                  "[ApparentHorizons][Unit]") {
  // Set up Kerr horizon
  constexpr int l_max = 20;
  const double mass = 2.0;
  const std::array<double, 3> spin{{0.04, 0.08, 0.12}};
  const auto spin_magnitude = magnitude(spin);
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  const auto horizon_radius = gr::Solutions::kerr_horizon_radius(
      Strahlkorper<Frame::Inertial>(l_max, l_max, 2.0, center)
          .ylm_spherepack()
          .theta_phi_points(),
      mass, spin);
  const auto kerr_horizon =
      Strahlkorper<Frame::Inertial>(l_max, l_max, get(horizon_radius), center);

  const auto horizon_radius_with_spin_on_z_axis =
      gr::Solutions::kerr_horizon_radius(
          Strahlkorper<Frame::Inertial>(l_max, l_max, 2.0, center)
              .ylm_spherepack()
              .theta_phi_points(),
          mass, {{0.0, 0.0, spin_magnitude}});
  const auto kerr_horizon_with_spin_on_z_axis = Strahlkorper<Frame::Inertial>(
      l_max, l_max, get(horizon_radius_with_spin_on_z_axis), center);

  // Check that the StrahlkorperGr::spin_vector() correctly recovers the
  // chosen dimensionless spin
  test_spin_vector(gr::Solutions::KerrSchild{mass, spin, center}, kerr_horizon,
                   mass, spin, horizon_radius_with_spin_on_z_axis,
                   kerr_horizon_with_spin_on_z_axis.ylm_spherepack());
}
