// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <unordered_map>

#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
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

  const gr::Solutions::KerrSchild solution(mass, spin, {0.0, 0.0, 0.0});
  const auto solution_vars = solution.variables(
      surface_coords, time.id,
      typename gr::Solutions::KerrSchild::tags<DataVector, Frame::Inertial>{});

  std::unordered_map<ElementId<3>,
                     ah::Storage::VolumeVariables<Frame::Inertial>>
      all_volume_variables{};

  // Check that if there aren't any volume variables, that nothing happens
  ah::interpolate_volume_data(make_not_null(&current_iteration),
                              make_not_null(&all_volume_variables), time,
                              domain, functions_of_time);
  CHECK(current_iteration.interpolation_is_done_for_these_elements.empty());
  CHECK(current_iteration.indicies_interpolated_to_thus_far.empty());
  CHECK(current_iteration.interpolated_vars.number_of_grid_points() == 0_st);

  size_t num_previous_indices_interpolated_to = 0;
  for (const auto& element_id : element_ids) {
    const Mesh mesh{domain_creator.initial_extents()[element_id.block_id()],
                    Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto};

    all_volume_variables[element_id].source_vars = compute_source_vars(
        solution, time, element_id, blocks[element_id.block_id()], mesh);
    all_volume_variables[element_id].mesh = mesh;

    ah::interpolate_volume_data(make_not_null(&current_iteration),
                                make_not_null(&all_volume_variables), time,
                                domain, functions_of_time);

    // Check that we finished interpolation and that the points we interpolated
    // to aren't the default fill value
    CHECK(current_iteration.interpolation_is_done_for_these_elements.contains(
        element_id));
    // We could in theory figure out which points are in which element for a
    // given l_max, but that's quite tedious and we don't need such a stringent
    // test
    CHECK_FALSE(current_iteration.indicies_interpolated_to_thus_far.size() ==
                num_previous_indices_interpolated_to);
    num_previous_indices_interpolated_to =
        current_iteration.indicies_interpolated_to_thus_far.size();
    tmpl::for_each<ah::vars_to_interpolate_to_target<3, Frame::Inertial>>(
        [&]<typename Tag>(tmpl::type_<Tag>) {
          auto& interpolated_var =
              get<Tag>(current_iteration.interpolated_vars);
          for (size_t j = 0; j < interpolated_var.size(); j++) {
            for (const size_t index :
                 current_iteration.indicies_interpolated_to_thus_far) {
              CHECK(interpolated_var[j][index] !=
                    std::numeric_limits<double>::max());
            }
          }
        });
  }

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

  // Test sending volume data again that it doesn't do anything
  const auto& element_id = element_ids[0];
  const Mesh mesh{domain_creator.initial_extents()[element_id.block_id()],
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto};

  ah::interpolate_volume_data(make_not_null(&current_iteration),
                              make_not_null(&all_volume_variables), time,
                              domain, functions_of_time);

  // Check interpolation is still done
  CHECK(current_iteration.interpolation_is_done_for_these_elements.contains(
      element_id));
  // This shouldn't have changed
  CHECK(current_iteration.indicies_interpolated_to_thus_far.size() ==
        num_previous_indices_interpolated_to);
  // Again check that all points are interpolated to
  check_no_max();
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.InterpolateVolumeVars",
                  "[ApparentHorizonFinder][Unit]") {
  // We don't need time dependent maps for this test. The actual time dependent
  // parts are tested elsewhere
  test_interpolate_volume_vars();
}
}  // namespace
