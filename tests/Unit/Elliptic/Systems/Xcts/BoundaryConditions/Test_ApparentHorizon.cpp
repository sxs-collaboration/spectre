// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/KerrHorizonConforming.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/ApparentHorizon.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::BoundaryConditions {

namespace {
// Make a metric approximately Riemannian
void make_metric_riemannian(
    const gsl::not_null<tnsr::II<DataVector, 3>*> inv_conformal_metric) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < i; ++j) {
      inv_conformal_metric->get(i, j) *= 1.e-2;
    }
    inv_conformal_metric->get(i, i) = abs(inv_conformal_metric->get(i, i));
  }
}

// Generate a face normal pretending the surface is a coordinate sphere
std::tuple<tnsr::i<DataVector, 3>, tnsr::ij<DataVector, 3>, Scalar<DataVector>>
make_spherical_face_normal(
    tnsr::I<DataVector, 3> x, const std::array<double, 3>& center,
    const tnsr::II<DataVector, 3>& inv_conformal_metric) {
  for (size_t d = 0; d < 3; ++d) {
    x.get(d) -= gsl::at(center, d);
  }
  Scalar<DataVector> proper_radius{x.begin()->size(), 0.};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(proper_radius) +=
          inv_conformal_metric.get(i, j) * x.get(i) * x.get(j);
    }
  }
  get(proper_radius) = sqrt(get(proper_radius));
  tnsr::i<DataVector, 3> face_normal{x.begin()->size()};
  get<0>(face_normal) = -get<0>(x) / get(proper_radius);
  get<1>(face_normal) = -get<1>(x) / get(proper_radius);
  get<2>(face_normal) = -get<2>(x) / get(proper_radius);
  tnsr::ij<DataVector, 3> deriv_unnormalized_face_normal{x.begin()->size(), 0.};
  get<0, 0>(deriv_unnormalized_face_normal) = -1.;
  get<1, 1>(deriv_unnormalized_face_normal) = -1.;
  get<2, 2>(deriv_unnormalized_face_normal) = -1.;
  return {std::move(face_normal), std::move(deriv_unnormalized_face_normal),
          std::move(proper_radius)};
}
std::tuple<tnsr::i<DataVector, 3>, tnsr::ij<DataVector, 3>, Scalar<DataVector>>
make_spherical_face_normal_flat_cartesian(tnsr::I<DataVector, 3> x,
                                          const std::array<double, 3>& center) {
  for (size_t d = 0; d < 3; ++d) {
    x.get(d) -= gsl::at(center, d);
  }
  Scalar<DataVector> euclidean_radius = magnitude(x);
  tnsr::i<DataVector, 3> face_normal{x.begin()->size()};
  get<0>(face_normal) = -get<0>(x) / get(euclidean_radius);
  get<1>(face_normal) = -get<1>(x) / get(euclidean_radius);
  get<2>(face_normal) = -get<2>(x) / get(euclidean_radius);
  tnsr::ij<DataVector, 3> deriv_unnormalized_face_normal{x.begin()->size(), 0.};
  get<0, 0>(deriv_unnormalized_face_normal) = -1.;
  get<1, 1>(deriv_unnormalized_face_normal) = -1.;
  get<2, 2>(deriv_unnormalized_face_normal) = -1.;
  return {std::move(face_normal), std::move(deriv_unnormalized_face_normal),
          std::move(euclidean_radius)};
}

template <Xcts::Geometry ConformalGeometry, bool Linearized, typename... Args>
void apply_boundary_condition_impl(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    Scalar<DataVector> conformal_factor,
    Scalar<DataVector> lapse_times_conformal_factor,
    tnsr::I<DataVector, 3> n_dot_longitudinal_shift_excess,
    const std::array<double, 3>& center, const std::array<double, 3>& rotation,
    Args&&... args) {
  const ApparentHorizon<ConformalGeometry> boundary_condition{
      center, rotation, std::nullopt, std::nullopt};
  const auto direction = Direction<3>::lower_xi();
  const auto box = db::create<domain::make_faces_tags<
      3,
      tmpl::conditional_t<
          Linearized,
          typename ApparentHorizon<ConformalGeometry>::argument_tags_linearized,
          typename ApparentHorizon<ConformalGeometry>::argument_tags>,
      tmpl::conditional_t<
          Linearized,
          typename ApparentHorizon<ConformalGeometry>::volume_tags_linearized,
          typename ApparentHorizon<ConformalGeometry>::volume_tags>>>(
      DirectionMap<3, std::decay_t<decltype(args)>>{
          {direction, std::move(args)}}...);
  elliptic::apply_boundary_condition<Linearized>(
      boundary_condition, box, direction, make_not_null(&conformal_factor),
      make_not_null(&lapse_times_conformal_factor), shift_excess,
      n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      make_not_null(&n_dot_longitudinal_shift_excess));
}

void apply_boundary_condition(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    const std::array<double, 3>& center, const std::array<double, 3>& rotation,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    tnsr::II<DataVector, 3> inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  make_metric_riemannian(make_not_null(&inv_conformal_metric));
  auto [face_normal, deriv_unnormalized_face_normal, face_normal_magnitude] =
      make_spherical_face_normal(x, center, inv_conformal_metric);
  apply_boundary_condition_impl<Xcts::Geometry::Curved, false>(
      n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient, shift_excess,
      conformal_factor, lapse_times_conformal_factor,
      n_dot_longitudinal_shift_excess, center, rotation, std::move(face_normal),
      std::move(deriv_unnormalized_face_normal),
      std::move(face_normal_magnitude), x, extrinsic_curvature_trace,
      shift_background, longitudinal_shift_background,
      std::move(inv_conformal_metric), conformal_christoffel_second_kind);
}

void apply_boundary_condition_flat_cartesian(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    const std::array<double, 3>& center, const std::array<double, 3>& rotation,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background) {
  auto [face_normal, deriv_unnormalized_face_normal, face_normal_magnitude] =
      make_spherical_face_normal_flat_cartesian(x, center);
  apply_boundary_condition_impl<Xcts::Geometry::FlatCartesian, false>(
      n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient, shift_excess,
      conformal_factor, lapse_times_conformal_factor,
      n_dot_longitudinal_shift_excess, center, rotation, std::move(face_normal),
      std::move(deriv_unnormalized_face_normal),
      std::move(face_normal_magnitude), x, extrinsic_curvature_trace,
      shift_background, longitudinal_shift_background);
}

void apply_boundary_condition_linearized(
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess_correction,
    const std::array<double, 3>& center, const std::array<double, 3>& rotation,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    tnsr::II<DataVector, 3> inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  make_metric_riemannian(make_not_null(&inv_conformal_metric));
  auto [face_normal, deriv_unnormalized_face_normal, face_normal_magnitude] =
      make_spherical_face_normal(x, center, inv_conformal_metric);
  apply_boundary_condition_impl<Xcts::Geometry::Curved, true>(
      n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      shift_excess_correction, conformal_factor_correction,
      lapse_times_conformal_factor_correction,
      n_dot_longitudinal_shift_excess_correction, center, rotation,
      std::move(face_normal), std::move(deriv_unnormalized_face_normal),
      std::move(face_normal_magnitude), x, extrinsic_curvature_trace,
      longitudinal_shift_background, conformal_factor,
      lapse_times_conformal_factor, n_dot_longitudinal_shift_excess,
      std::move(inv_conformal_metric), conformal_christoffel_second_kind);
}

void apply_boundary_condition_linearized_flat_cartesian(
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess_correction,
    const std::array<double, 3>& center, const std::array<double, 3>& rotation,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess) {
  auto [face_normal, deriv_unnormalized_face_normal, face_normal_magnitude] =
      make_spherical_face_normal_flat_cartesian(x, center);
  apply_boundary_condition_impl<Xcts::Geometry::FlatCartesian, true>(
      n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      shift_excess_correction, conformal_factor_correction,
      lapse_times_conformal_factor_correction,
      n_dot_longitudinal_shift_excess_correction, center, rotation,
      std::move(face_normal), std::move(deriv_unnormalized_face_normal),
      std::move(face_normal_magnitude), x, extrinsic_curvature_trace,
      longitudinal_shift_background, conformal_factor,
      lapse_times_conformal_factor, n_dot_longitudinal_shift_excess);
}

void test_creation(
    const std::array<double, 3>& center, const std::array<double, 3>& rotation,
    const std::optional<gr::Solutions::KerrSchild>& kerr_solution_for_lapse,
    const std::optional<gr::Solutions::KerrSchild>&
        kerr_solution_for_negative_expansion,
    const std::string& option_string) {
  INFO("Test factory-creation");
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<elliptic::BoundaryConditions::BoundaryCondition<
          3, tmpl::list<Registrars::ApparentHorizon<Xcts::Geometry::Curved>>>>>(
      option_string);
  REQUIRE(dynamic_cast<const ApparentHorizon<Xcts::Geometry::Curved>*>(
              created.get()) != nullptr);
  const auto& boundary_condition =
      dynamic_cast<const ApparentHorizon<Xcts::Geometry::Curved>&>(*created);
  {
    INFO("Properties");
    CHECK(boundary_condition.center() == center);
    CHECK(boundary_condition.rotation() == rotation);
    // Work around an issue where Catch2 needs a stream operator to check
    // optionals are equal
    CHECK(boundary_condition.kerr_solution_for_lapse().has_value() ==
          kerr_solution_for_lapse.has_value());
    if (kerr_solution_for_lapse.has_value()) {
      CHECK(*boundary_condition.kerr_solution_for_lapse() ==
            *kerr_solution_for_lapse);
    }
    CHECK(
        boundary_condition.kerr_solution_for_negative_expansion().has_value() ==
        kerr_solution_for_negative_expansion.has_value());
    if (kerr_solution_for_negative_expansion.has_value()) {
      CHECK(*boundary_condition.kerr_solution_for_negative_expansion() ==
            *kerr_solution_for_negative_expansion);
    }
  }
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
}

void test_with_random_values() {
  INFO("Random-value tests");
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Elliptic/Systems/Xcts/BoundaryConditions/");
  pypp::check_with_random_values<11>(
      &apply_boundary_condition, "ApparentHorizon",
      {"normal_dot_conformal_factor_gradient",
       "normal_dot_lapse_times_conformal_factor_gradient", "shift_excess"},
      {{{0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      DataVector{3});
  pypp::check_with_random_values<9>(
      &apply_boundary_condition_flat_cartesian, "ApparentHorizon",
      {"normal_dot_conformal_factor_gradient_flat_cartesian",
       "normal_dot_lapse_times_conformal_factor_gradient",
       "shift_excess_flat_cartesian"},
      {{{0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      DataVector{3});
  pypp::check_with_random_values<13>(
      &apply_boundary_condition_linearized, "ApparentHorizon",
      {"normal_dot_conformal_factor_gradient_correction",
       "normal_dot_lapse_times_conformal_factor_gradient",
       "shift_excess_correction"},
      {{{0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      DataVector{3});
  pypp::check_with_random_values<11>(
      &apply_boundary_condition_linearized_flat_cartesian, "ApparentHorizon",
      {"normal_dot_conformal_factor_gradient_correction_flat_cartesian",
       "normal_dot_lapse_times_conformal_factor_gradient",
       "shift_excess_correction_flat_cartesian"},
      {{{0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.}}},
      DataVector{3});
}

void test_consistency_with_kerr(const bool compute_expansion) {
  INFO("Consistency with Kerr solution");
  CAPTURE(compute_expansion);
  const double mass = 1.;
  const std::array<double, 3> center{{0., 0., 0.}};
  const std::array<double, 3> dimensionless_spin{{0., 0., 0.8}};
  const double horizon_kerrschild_radius =
      mass * (1. + sqrt(1. - dot(dimensionless_spin, dimensionless_spin)));
  CAPTURE(horizon_kerrschild_radius);
  // Eq. (8) in https://arxiv.org/abs/1506.01689
  std::array<double, 3> rotation =
      -0.5 * dimensionless_spin / horizon_kerrschild_radius;
  CAPTURE(rotation);
  const Solutions::Kerr<> solution{mass, dimensionless_spin, {{0., 0., 0.}}};
  const ApparentHorizon<Xcts::Geometry::Curved> kerr_horizon{
      center, rotation, solution,
      // Check with and without the negative-expansion condition. Either the
      // expansion is _computed_ to be zero from the Kerr solution at the
      // horizon, or it is just set to zero.
      compute_expansion ? std::make_optional(solution) : std::nullopt};

  // Set up a wedge with a Kerr-horizon-conforming inner surface
  const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  const domain::CoordinateMaps::Wedge<3> wedge_map{
      horizon_kerrschild_radius,
      2. * horizon_kerrschild_radius,
      1.,
      1.,
      {},
      false};
  const domain::CoordinateMaps::KerrHorizonConforming horizon_map{
      dimensionless_spin};
  const auto coord_map =
      domain::make_coordinate_map_base<Frame::ElementLogical, Frame::Inertial>(
          wedge_map, horizon_map);
  // Set up a mesh so we can numerically differentiate the Jacobian
  const auto logical_coords = logical_coordinates(mesh);
  const tnsr::I<DataVector, 3> inertial_coords = (*coord_map)(logical_coords);
  const auto inv_jacobian = coord_map->inv_jacobian(logical_coords);
  using inv_jac_tag =
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>;
  Variables<tmpl::list<inv_jac_tag>> vars_to_derive{num_points};
  get<inv_jac_tag>(vars_to_derive) = inv_jacobian;
  const auto deriv_vars = partial_derivatives<tmpl::list<inv_jac_tag>>(
      vars_to_derive, mesh, inv_jacobian);
  const auto& deriv_inv_jac =
      get<::Tags::deriv<inv_jac_tag, tmpl::size_t<3>, Frame::Inertial>>(
          deriv_vars);
  // Coords on the face
  const auto direction = Direction<3>::lower_zeta();
  const size_t slice_index = index_to_slice_at(mesh.extents(), direction);
  const auto x = data_on_slice(inertial_coords, mesh.extents(),
                               direction.dimension(), slice_index);
  // Get background fields from the solution
  const auto background_fields = solution.variables(
      x,
      tmpl::list<
          Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
          Tags::ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>,
          gr::Tags::TraceExtrinsicCurvature<DataVector>,
          Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
          Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataVector, 3, Frame::Inertial>>{});
  const auto& inv_conformal_metric =
      get<Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
          background_fields);
  // Set up the face normal on the horizon surface. It points _into_ the
  // horizon because the computational domain fills the space outside of it
  // and the normal always points away from the computational domain.
  const Mesh<2> face_mesh = mesh.slice_away(direction.dimension());
  const size_t face_num_points = face_mesh.number_of_grid_points();
  auto face_normal = unnormalized_face_normal(face_mesh, *coord_map, direction);
  const auto face_normal_magnitude =
      magnitude(face_normal, inv_conformal_metric);
  for (size_t d = 0; d < 3; ++d) {
    face_normal.get(d) /= get(face_normal_magnitude);
  }
  tnsr::ij<DataVector, 3> deriv_unnormalized_face_normal{face_num_points};
  const auto deriv_inv_jac_on_face = data_on_slice(
      deriv_inv_jac, mesh.extents(), direction.dimension(), slice_index);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      deriv_unnormalized_face_normal.get(i, j) =
          direction.sign() *
          deriv_inv_jac_on_face.get(i, direction.dimension(), j);
    }
  }

  // Retrieve the expected surface vars and fluxes from the solution
  const auto surface_vars_expected =
      variables_from_tagged_tuple(solution.variables(
          x, tmpl::list<Tags::ConformalFactor<DataVector>,
                        Tags::LapseTimesConformalFactor<DataVector>,
                        Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>{}));
  const auto surface_fluxes_expected =
      variables_from_tagged_tuple(solution.variables(
          x,
          tmpl::list<::Tags::Flux<Tags::ConformalFactor<DataVector>,
                                  tmpl::size_t<3>, Frame::Inertial>,
                     ::Tags::Flux<Tags::LapseTimesConformalFactor<DataVector>,
                                  tmpl::size_t<3>, Frame::Inertial>,
                     Tags::LongitudinalShiftExcess<DataVector, 3,
                                                   Frame::Inertial>>{}));
  Variables<tmpl::list<
      ::Tags::NormalDotFlux<Tags::ConformalFactor<DataVector>>,
      ::Tags::NormalDotFlux<Tags::LapseTimesConformalFactor<DataVector>>,
      ::Tags::NormalDotFlux<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>>>
      n_dot_surface_fluxes_expected{num_points};
  normal_dot_flux(make_not_null(&n_dot_surface_fluxes_expected), face_normal,
                  surface_fluxes_expected);
  // Apply the boundary conditions, passing garbage for the data that the
  // boundary conditions are expected to fill
  auto surface_vars = surface_vars_expected;
  auto n_dot_surface_fluxes = n_dot_surface_fluxes_expected;
  get(get<::Tags::NormalDotFlux<Tags::ConformalFactor<DataVector>>>(
      n_dot_surface_fluxes)) = std::numeric_limits<double>::signaling_NaN();
  get<0>(get<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(surface_vars)) =
      std::numeric_limits<double>::signaling_NaN();
  get<1>(get<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(surface_vars)) =
      std::numeric_limits<double>::signaling_NaN();
  get<2>(get<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(surface_vars)) =
      std::numeric_limits<double>::signaling_NaN();
  kerr_horizon.apply(
      make_not_null(&get<Tags::ConformalFactor<DataVector>>(surface_vars)),
      make_not_null(
          &get<Tags::LapseTimesConformalFactor<DataVector>>(surface_vars)),
      make_not_null(&get<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
          surface_vars)),
      make_not_null(
          &get<::Tags::NormalDotFlux<Tags::ConformalFactor<DataVector>>>(
              n_dot_surface_fluxes)),
      make_not_null(&get<::Tags::NormalDotFlux<
                        Tags::LapseTimesConformalFactor<DataVector>>>(
          n_dot_surface_fluxes)),
      make_not_null(&get<::Tags::NormalDotFlux<
                        Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>>(
          n_dot_surface_fluxes)),
      face_normal, deriv_unnormalized_face_normal, face_normal_magnitude, x,
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(background_fields),
      get<Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>(
          background_fields),
      get<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>(background_fields),
      get<Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
          background_fields),
      get<Tags::ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>>(
          background_fields));
  // Check the result.
  CHECK_VARIABLES_APPROX(surface_vars, surface_vars_expected);
  CHECK_VARIABLES_APPROX(n_dot_surface_fluxes, n_dot_surface_fluxes_expected);
}

void test_parse_errors() {
  CHECK_THROWS_WITH(
      ApparentHorizon<Xcts::Geometry::Curved>(
          {{0., 1., 2.}},
          {{0.1, 0.2, 0.3}},
          {{2.3, {{0.4, 0.5, 0.6}}, {{0.5, 0., 0.}}}},
          std::nullopt,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("The Kerr solution supplied to the 'Lapse' "
                                "option has a non-zero 'Center'."));
  CHECK_THROWS_WITH(
      ApparentHorizon<Xcts::Geometry::Curved>(
          {{0., 1., 2.}},
          {{0.1, 0.2, 0.3}},
          std::nullopt,
          {{2.3, {{0.4, 0.5, 0.6}}, {{0.5, 0., 0.}}}},
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The Kerr solution supplied to the 'NegativeExpansion' "
          "option has a non-zero 'Center'."));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Xcts.BoundaryConditions.ApparentHorizon",
                  "[Unit][Elliptic]") {
  test_creation({{1., 2., 3.}}, {{0.1, 0.2, 0.3}}, std::nullopt, std::nullopt,
                "ApparentHorizon:\n"
                "  Center: [1., 2., 3.]\n"
                "  Rotation: [0.1, 0.2, 0.3]\n"
                "  Lapse: Auto\n"
                "  NegativeExpansion: None\n");
  test_creation({{1., 2., 3.}}, {{0.1, 0.2, 0.3}},
                {{2.3, {{0.4, 0.5, 0.6}}, {{0., 0., 0.}}}},
                {{3.4, {{0.3, 0.2, 0.1}}, {{0., 0., 0.}}}},
                "ApparentHorizon:\n"
                "  Center: [1., 2., 3.]\n"
                "  Rotation: [0.1, 0.2, 0.3]\n"
                "  Lapse:\n"
                "    Mass: 2.3\n"
                "    Spin: [0.4, 0.5, 0.6]\n"
                "    Center: [0., 0., 0.]\n"
                "  NegativeExpansion:\n"
                "    Mass: 3.4\n"
                "    Spin: [0.3, 0.2, 0.1]\n"
                "    Center: [0., 0., 0.]\n");
  test_parse_errors();
  test_with_random_values();
  test_consistency_with_kerr(false);
  test_consistency_with_kerr(true);
}

}  // namespace Xcts::BoundaryConditions
