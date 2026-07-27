// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>

#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeVarsToInterpolateToTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename Fr>
void test_compute_horizon_volume_quantities(const bool is_time_dependent) {
  CAPTURE(is_time_dependent);
  CAPTURE(pretty_type::name<Fr>());
  const size_t number_of_grid_points = 12;

  const double mass = 1.0;
  const std::array spin{0.1, 0.2, 0.3};
  const LinkedMessageId<double> time{0.01, std::nullopt};
  // We have time dependent maps so the code takes the correct paths, but we
  // make them the identity so the vars that we interpolate are the same in each
  // frame which makes checking them much easier. Also we have small grid
  // spacing so derivatives are more accurate.
  const domain::creators::Sphere domain_creator{
      0.9,
      1.0,
      domain::creators::Sphere::Excision{nullptr},
      1_st,
      number_of_grid_points,
      true,
      std::nullopt,
      {0.95},
      domain::CoordinateMaps::Distribution::Linear,
      ShellWedges::All,
      is_time_dependent
          ? std::optional{domain::creators::sphere::TimeDependentMapOptions{
                0.0,
                domain::creators::time_dependent_options::ShapeMapOptions<
                    false, ::domain::ObjectLabel::None>{8_st, std::nullopt},
                std::nullopt, std::nullopt,
                domain::creators::time_dependent_options::TranslationMapOptions<
                    3>{std::array{std::array{0.0, 0.0, 0.0},
                                  std::array{0.0, 0.0, 0.0},
                                  std::array{0.0, 0.0, 0.0}}},
                true, std::nullopt}}
          : std::nullopt};
  const auto domain = domain_creator.create_domain();
  const auto& functions_of_time = domain_creator.functions_of_time();

  // Just use the first block, it has all maps and all frames
  const auto& block = domain.blocks()[0];
  const auto element_ids = initial_element_ids(
      block.id(), domain_creator.initial_refinement_levels()[block.id()]);

  // Set up target coordinates, and jacobian.
  const Mesh mesh{domain_creator.initial_extents()[block.id()],
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);
  tnsr::I<DataVector, 3, Fr> target_frame_coords{};
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{};
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inv_jacobian_logical_to_inertial{mesh.number_of_grid_points(), 0.0};
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Fr>
      inv_jacobian_logical_to_frame{mesh.number_of_grid_points(), 0.0};
  if (is_time_dependent) {
    const ElementMap<3, Frame::Grid> map_logical_to_grid{
        element_ids[0], block.moving_mesh_logical_to_grid_map().get_clone()};

    inertial_coords = block.moving_mesh_grid_to_inertial_map()(
        map_logical_to_grid(logical_coords), time.id, functions_of_time);

    const auto inv_jacobian_logical_to_grid =
        map_logical_to_grid.inv_jacobian(logical_coords);
    const auto inv_jacobian_grid_to_inertial =
        block.moving_mesh_grid_to_inertial_map().inv_jacobian(
            map_logical_to_grid(logical_coords), time.id, functions_of_time);

    inv_jacobian_logical_to_inertial = tenex::evaluate<ti::I, ti::j>(
        inv_jacobian_logical_to_grid(ti::I, ti::k) *
        inv_jacobian_grid_to_inertial(ti::K, ti::j));

    if constexpr (std::is_same_v<Fr, Frame::Grid>) {
      inv_jacobian_logical_to_frame = inv_jacobian_logical_to_grid;
    } else if constexpr (std::is_same_v<Fr, Frame::Inertial>) {
      inv_jacobian_logical_to_frame = inv_jacobian_logical_to_inertial;
    } else {
      const auto inv_jacobian_grid_to_distorted =
          block.moving_mesh_grid_to_distorted_map().inv_jacobian(
              map_logical_to_grid(logical_coords), time.id, functions_of_time);

      inv_jacobian_logical_to_frame = tenex::evaluate<ti::I, ti::j>(
          inv_jacobian_logical_to_grid(ti::I, ti::k) *
          inv_jacobian_grid_to_distorted(ti::K, ti::j));
    }

    if constexpr (std::is_same_v<Fr, Frame::Grid>) {
      target_frame_coords = map_logical_to_grid(logical_coords);
    } else if constexpr (std::is_same_v<Fr, Frame::Distorted>) {
      target_frame_coords = block.moving_mesh_grid_to_distorted_map()(
          map_logical_to_grid(logical_coords), time.id, functions_of_time);
    } else {
      static_assert(std::is_same_v<Fr, Frame::Inertial>,
                    "Fr must be the Inertial frame");
      target_frame_coords = block.moving_mesh_grid_to_inertial_map()(
          map_logical_to_grid(logical_coords), time.id, functions_of_time);
    }

  } else {
    // time-independent.
    if (std::is_same_v<Fr, Frame::Distorted>) {
      ERROR("Can't have a time independent distorted frame.");
    }
    // Grid == inertial so just get inertial and set the frame
    const ElementMap<3, Frame::Inertial> map_logical_to_inertial{
        element_ids[0], block.stationary_map().get_clone()};
    inv_jacobian_logical_to_inertial =
        map_logical_to_inertial.inv_jacobian(logical_coords);

    inertial_coords = map_logical_to_inertial(logical_coords);
    for (size_t i = 0; i < 3; i++) {
      target_frame_coords[i] = inertial_coords[i];
      for (size_t j = 0; j < 3; j++) {
        inv_jacobian_logical_to_frame.get(i, j) =
            inv_jacobian_logical_to_inertial.get(i, j);
      }
    }
  }

  // Set up analytic solution.
  const gr::Solutions::KerrSchild solution(mass, spin, {0.0, 0.0, 0.0});
  const auto solution_vars_inertial_frame = solution.variables(
      inertial_coords, time.id,
      typename gr::Solutions::KerrSchild::tags<DataVector, Frame::Inertial>{});

  Variables<ah::source_vars<3>> source_vars{};
  Variables<ah::vars_to_interpolate_to_target<3, Fr>> target_vars{};
  // Set g, pi, and phi in the inertial frame
  {
    const auto& lapse =
        get<gr::Tags::Lapse<DataVector>>(solution_vars_inertial_frame);
    const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataVector>>>(
        solution_vars_inertial_frame);
    const auto& d_lapse =
        get<typename gr::Solutions::KerrSchild::DerivLapse<DataVector>>(
            solution_vars_inertial_frame);
    const auto& shift =
        get<gr::Tags::Shift<DataVector, 3>>(solution_vars_inertial_frame);
    const auto& d_shift =
        get<typename gr::Solutions::KerrSchild::DerivShift<DataVector>>(
            solution_vars_inertial_frame);
    const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(
        solution_vars_inertial_frame);
    const auto& spatial_metric = get<gr::Tags::SpatialMetric<DataVector, 3>>(
        solution_vars_inertial_frame);
    const auto& dt_spatial_metric =
        get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(
            solution_vars_inertial_frame);
    const auto& d_spatial_metric =
        get<typename gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
            solution_vars_inertial_frame);

    source_vars.initialize(get(lapse).size(), 0.0);
    get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(source_vars) =
        gr::spacetime_metric(lapse, shift, spatial_metric);
    get<::gh::Tags::Phi<DataVector, 3>>(source_vars) = gh::phi(
        lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);
    get<::gh::Tags::Pi<DataVector, 3>>(source_vars) = gh::pi(
        lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric,
        get<::gh::Tags::Phi<DataVector, 3>>(source_vars));

    // Need to compute numerical deriv of Phi.
    get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                    Frame::Inertial>>(source_vars) =
        partial_derivative(get<::gh::Tags::Phi<DataVector, 3>>(source_vars),
                           mesh, inv_jacobian_logical_to_inertial);
  }

  // Compute other vars
  ah::compute_vars_to_interpolate_to_target(
      make_not_null(&target_vars),
      get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(source_vars),
      get<::gh::Tags::Pi<DataVector, 3>>(source_vars),
      get<::gh::Tags::Phi<DataVector, 3>>(source_vars),
      get<Tags::deriv<::gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                      Frame::Inertial>>(source_vars),
      time, domain, mesh, element_ids[0], functions_of_time);

  // Now make sure those computed vars are correct.
  const auto solution_vars_target_frame = solution.variables(
      target_frame_coords, 0.0,
      tmpl::pop_back<ah::vars_to_interpolate_to_target<3, Fr>>{});
  // Expected vars
  const auto& expected_spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3, Fr>>(
          solution_vars_target_frame);
  const auto& expected_inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3, Fr>>(
          solution_vars_target_frame);
  const auto& expected_extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3, Fr>>(
          solution_vars_target_frame);
  const auto& expected_christoffel =
      get<gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, Fr>>(
          solution_vars_target_frame);
  const auto deriv_christoffel = partial_derivative(
      expected_christoffel, mesh, inv_jacobian_logical_to_frame);
  const auto expected_ricci =
      gr::ricci_tensor(expected_christoffel, deriv_christoffel);

  // Computed vars
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3, Fr>>(target_vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3, Fr>>(target_vars);
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3, Fr>>(target_vars);
  const auto& christoffel =
      get<gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, Fr>>(
          target_vars);
  const auto& ricci =
      get<gr::Tags::SpatialRicci<DataVector, 3, Fr>>(target_vars);

  CHECK_ITERABLE_APPROX(expected_spatial_metric, spatial_metric);
  CHECK_ITERABLE_APPROX(expected_inverse_spatial_metric,
                        inverse_spatial_metric);
  CHECK_ITERABLE_APPROX(expected_extrinsic_curvature, extrinsic_curvature);
  // Larger tolerance because derivatives are inaccurate
  const Approx custom_approx_1 = Approx::custom().epsilon(1.e-6).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_christoffel, christoffel,
                               custom_approx_1);
  const Approx custom_approx_2 = Approx::custom().epsilon(1.e-5).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_ricci, ricci, custom_approx_2);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.ComputeVarsToInterpolateToTarget",
                  "[ApparentHorizonFinder][Unit]") {
  // time-independent.
  test_compute_horizon_volume_quantities<Frame::Grid>(false);
  test_compute_horizon_volume_quantities<Frame::Inertial>(false);

  // time-dependent.
  test_compute_horizon_volume_quantities<Frame::Grid>(true);
  test_compute_horizon_volume_quantities<Frame::Distorted>(true);
  test_compute_horizon_volume_quantities<Frame::Inertial>(true);
}
}  // namespace
