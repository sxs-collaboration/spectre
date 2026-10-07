// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>

#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeVarsToInterpolateToTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolateVolumeVars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {
Variables<ah::source_vars<3>> compute_source_vars(
    const gr::Solutions::KerrSchild& solution,
    const LinkedMessageId<double>& time, const ElementId<3>& element_id,
    const Block<3>& block, const Mesh<3>& mesh) {
  const auto logical_coords = logical_coordinates(mesh);
  const ElementMap<3, Frame::Inertial> map_logical_to_inertial{
      element_id, block.stationary_map().get_clone()};
  const auto inertial_coords = map_logical_to_inertial(logical_coords);

  const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inv_jacobian_logical_to_inertial =
          map_logical_to_inertial.inv_jacobian(logical_coords);

  const auto solution_vars = solution.variables(
      inertial_coords, time.id,
      typename gr::Solutions::KerrSchild::tags<DataVector, Frame::Inertial>{});

  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
  const auto& dt_lapse =
      get<Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
  const auto& d_lapse =
      get<typename gr::Solutions::KerrSchild::DerivLapse<DataVector>>(
          solution_vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(solution_vars);
  const auto& d_shift =
      get<typename gr::Solutions::KerrSchild::DerivShift<DataVector>>(
          solution_vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(solution_vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(solution_vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(solution_vars);
  const auto& d_spatial_metric =
      get<typename gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
          solution_vars);

  Variables<ah::source_vars<3>> result{get(lapse).size()};
  get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(result) =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  get<::gh::Tags::Phi<DataVector, 3>>(result) =
      gh::phi(lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);
  get<::gh::Tags::Pi<DataVector, 3>>(result) =
      gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
             dt_spatial_metric, get<::gh::Tags::Phi<DataVector, 3>>(result));

  // Need to compute numerical deriv of Phi.
  get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                  Frame::Inertial>>(result) =
      partial_derivative(get<::gh::Tags::Phi<DataVector, 3>>(result), mesh,
                         inv_jacobian_logical_to_inertial);

  return result;
}

void test_interpolate_volume_vars() {
  const size_t number_of_grid_points = 8;

  const double mass = 1.0;
  const std::array spin{0.1, 0.2, 0.3};
  const std::array center{-0.01, -0.02, -0.03};
  const LinkedMessageId<double> time{0.01, std::nullopt};
  const domain::creators::Sphere domain_creator{
      0.7,
      6.0,
      domain::creators::Sphere::Excision{nullptr},
      0_st,
      number_of_grid_points,
      true};
  const auto domain = domain_creator.create_domain();
  const auto& functions_of_time = domain_creator.functions_of_time();

  const auto& blocks = domain.blocks();
  const auto element_ids = [&]() {
    std::vector<ElementId<3>> result{};
    for (const auto& block : blocks) {
      const auto temp_element_ids = initial_element_ids(
          block.id(), domain_creator.initial_refinement_levels()[block.id()]);
      result.insert(result.end(), temp_element_ids.begin(),
                    temp_element_ids.end());
    }
    return result;
  }();

  // For a quick test
  const size_t l_max = 6;
  const double radius = 2.0;

  ah::Storage::Iteration<Frame::Inertial> current_iteration{};
  current_iteration.strahlkorper =
      ylm::Strahlkorper<Frame::Inertial>{l_max, radius, center};
  const auto surface_coords =
      ylm::cartesian_coords(current_iteration.strahlkorper);
  current_iteration.block_coord_holders = ::block_logical_coordinates(
      domain, surface_coords, time.id, functions_of_time);
  const size_t expected_num_points =
      current_iteration.block_coord_holders->size();

  const gr::Solutions::KerrSchild solution(mass, spin, {0.0, 0.0, 0.0});
  const auto solution_vars = solution.variables(
      surface_coords, time.id,
      typename gr::Solutions::KerrSchild::tags<DataVector, Frame::Inertial>{});

  std::unordered_map<ElementId<3>,
                     ah::Storage::VolumeVariables<Frame::Inertial>>
      all_volume_variables{};

  size_t num_previous_indices_interpolated_to = 0;
  for (const auto& element_id : element_ids) {
    const Mesh mesh{domain_creator.initial_extents()[element_id.block_id()],
                    Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto};

    const auto source_vars = compute_source_vars(
        solution, time, element_id, blocks[element_id.block_id()], mesh);

    auto& volume_vars = all_volume_variables[element_id];
    volume_vars.mesh = mesh;
    ah::compute_vars_to_interpolate_to_target(
        make_not_null(&volume_vars.vars_to_interpolate_to_target),
        get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(source_vars),
        get<::gh::Tags::Pi<DataVector, 3>>(source_vars),
        get<::gh::Tags::Phi<DataVector, 3>>(source_vars),
        get<Tags::deriv<::gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(source_vars),
        time, domain, mesh, element_id, functions_of_time);

    const bool interpolated_any_points = ah::interpolate_volume_data(
        make_not_null(&current_iteration), volume_vars, element_id);

    CHECK(current_iteration.intersecting_element_ids.contains(element_id) ==
          interpolated_any_points);

    // Check that we finished interpolation and that the points we interpolated
    // to aren't the default fill value
    // We could in theory figure out which points are in which element for a
    // given l_max, but that's quite tedious and we don't need such a stringent
    // test
    const auto num_indices_interpolated_to = static_cast<size_t>(
        alg::count_if(current_iteration.indices_interpolated_to_thus_far,
                      [](const bool filled) { return filled; }));
    CHECK(num_indices_interpolated_to > num_previous_indices_interpolated_to);
    num_previous_indices_interpolated_to = num_indices_interpolated_to;
    tmpl::for_each<ah::vars_to_interpolate_to_target<3, Frame::Inertial>>(
        [&]<typename Tag>(tmpl::type_<Tag>) {
          auto& interpolated_var =
              get<Tag>(current_iteration.interpolated_vars);
          for (size_t j = 0; j < interpolated_var.size(); j++) {
            for (size_t index = 0; index < expected_num_points; index++) {
              if (current_iteration.indices_interpolated_to_thus_far[index]) {
                CHECK(interpolated_var[j][index] !=
                      std::numeric_limits<double>::max());
              }
            }
          }
        });
  }
  CHECK(current_iteration.interpolation_is_complete());

  const auto check_no_max = [&]() {
    tmpl::for_each<ah::vars_to_interpolate_to_target<3, Frame::Inertial>>(
        [&]<typename Tag>(tmpl::type_<Tag>) {
          auto& interpolated_var =
              get<Tag>(current_iteration.interpolated_vars);
          for (size_t j = 0; j < interpolated_var.size(); j++) {
            CHECK(alg::none_of(interpolated_var[j], [](const double value) {
              return value == std::numeric_limits<double>::max();
            }));
          }
        });
  };

  // Check all points have been interpolated to
  check_no_max();
}

// Points on shared element boundaries must be assigned to the same element
// regardless of the order in which element data arrives. Otherwise, the
// interpolated values depend on the (nondeterministic) message order because
// DG fields are discontinuous across element boundaries.
void test_element_order_independence() {
  const LinkedMessageId<double> time{0.01, std::nullopt};
  // Refine the wedges so that there are element boundaries within each block,
  // in particular in the equatorial plane
  const domain::creators::Sphere domain_creator{
      0.7, 6.0, domain::creators::Sphere::Excision{nullptr}, 1_st, 5_st, true};
  const auto domain = domain_creator.create_domain();
  const auto& functions_of_time = domain_creator.functions_of_time();
  const auto& blocks = domain.blocks();
  std::vector<ElementId<3>> element_ids{};
  for (const auto& block : blocks) {
    const auto block_element_ids = initial_element_ids(
        block.id(), domain_creator.initial_refinement_levels()[block.id()]);
    element_ids.insert(element_ids.end(), block_element_ids.begin(),
                       block_element_ids.end());
  }

  // An even l_max places a ring of collocation points in the equatorial plane,
  // which lies on element boundaries
  const size_t l_max = 6;
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{
      l_max, 2.0, {{0.0, 0.0, 0.0}}};
  const auto surface_coords = ylm::cartesian_coords(strahlkorper);
  const auto block_coord_holders = ::block_logical_coordinates(
      domain, surface_coords, time.id, functions_of_time);

  // Sanity check that some points lie exactly on shared element boundaries
  size_t num_points_on_element_boundaries = 0;
  for (const auto& block_coord_holder : block_coord_holders) {
    size_t num_elements_touching_point = 0;
    for (const auto& element_id : element_ids) {
      if (element_id.block_id() == block_coord_holder->id.get_index() and
          element_logical_coordinates(block_coord_holder->data, element_id)
              .has_value()) {
        ++num_elements_touching_point;
      }
    }
    if (num_elements_touching_point > 1) {
      ++num_points_on_element_boundaries;
    }
  }
  REQUIRE(num_points_on_element_boundaries > 0);

  // Use a solution that is not smooth at the element boundaries at the
  // discrete level, i.e. interpolation from different elements gives
  // different results
  const gr::Solutions::KerrSchild solution(1.0, {{0.1, 0.2, 0.3}},
                                           {{0.0, 0.0, 0.0}});
  std::unordered_map<ElementId<3>,
                     ah::Storage::VolumeVariables<Frame::Inertial>>
      all_volume_variables{};
  for (const auto& element_id : element_ids) {
    const Mesh mesh{domain_creator.initial_extents()[element_id.block_id()],
                    Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto};
    const auto source_vars = compute_source_vars(
        solution, time, element_id, blocks[element_id.block_id()], mesh);
    auto& volume_vars = all_volume_variables[element_id];
    volume_vars.mesh = mesh;
    ah::compute_vars_to_interpolate_to_target(
        make_not_null(&volume_vars.vars_to_interpolate_to_target),
        get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(source_vars),
        get<::gh::Tags::Pi<DataVector, 3>>(source_vars),
        get<::gh::Tags::Phi<DataVector, 3>>(source_vars),
        get<Tags::deriv<::gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(source_vars),
        time, domain, mesh, element_id, functions_of_time);
  }

  const auto interpolate_in_order =
      [&](const std::vector<ElementId<3>>& ordered_element_ids) {
        ah::Storage::Iteration<Frame::Inertial> iteration{};
        iteration.strahlkorper = strahlkorper;
        iteration.block_coord_holders = block_coord_holders;
        for (const auto& element_id : ordered_element_ids) {
          ah::interpolate_volume_data(make_not_null(&iteration),
                                      all_volume_variables.at(element_id),
                                      element_id);
        }
        REQUIRE(iteration.interpolation_is_complete());
        return iteration;
      };
  const auto forward = interpolate_in_order(element_ids);
  auto reversed_element_ids = element_ids;
  std::reverse(reversed_element_ids.begin(), reversed_element_ids.end());
  const auto reversed = interpolate_in_order(reversed_element_ids);
  // Must be bitwise identical, not just approximately equal
  CHECK(forward.interpolated_vars == reversed.interpolated_vars);
  CHECK(forward.intersecting_element_ids == reversed.intersecting_element_ids);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.InterpolateVolumeVars",
                  "[ApparentHorizonFinder][Unit]") {
  // We don't need time dependent maps for this test. The actual time dependent
  // parts are tested elsewhere
  test_interpolate_volume_vars();
  test_element_order_independence();
}
}  // namespace
