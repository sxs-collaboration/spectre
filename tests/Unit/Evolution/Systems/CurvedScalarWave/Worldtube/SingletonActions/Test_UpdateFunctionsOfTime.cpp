// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/UpdateFunctionsOfTime.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonChare.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/CurvedScalarWave/Worldtube/TestHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

template <typename Metavariables>
struct MockWorldtubeSingleton {
  using metavariables = Metavariables;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
  static constexpr size_t Dim = metavariables::volume_dim;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              db::AddSimpleTags<::Tags::Time,
                                Tags::ParticlePositionVelocity<Dim>,
                                ::Tags::Next<::Tags::TimeStepId>,
                                Tags::WorldtubeRadius, Tags::ExpirationTime>,
              db::AddComputeTags<>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::UpdateFunctionsOfTime,
                     domain::Actions::CheckFunctionsOfTimeAreReady<Dim>>>>;
  using component_being_mocked = WorldtubeSingleton<Metavariables>;
};

template <size_t Dim>
struct MockMetavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<MockWorldtubeSingleton<MockMetavariables>>;
  using const_global_cache_tags =
      tmpl::list<Tags::ExcisionSphere<Dim>, domain::Tags::Domain<3>,
                 Tags::WorldtubeRadiusParameters,
                 Tags::BlackHoleRadiusParameters>;
};

void test_particle_motion() {
  static constexpr size_t Dim = 3;
  using metavars = MockMetavariables<Dim>;
  domain::creators::register_derived_with_charm();
  using worldtube_chare = MockWorldtubeSingleton<metavars>;
  const double initial_worldtube_radius = 0.456;

  const auto bco_domain_creator =
      TestHelpers::CurvedScalarWave::Worldtube::worldtube_binary_compact_object<
          true>(6., initial_worldtube_radius, 0.1);
  domain::FunctionsOfTime::register_derived_with_charm();

  const auto domain = bco_domain_creator->create_domain();
  const auto wt_excision_sphere =
      domain.excision_spheres().at("ExcisionSphereA");
  const auto bh_excision_sphere =
      domain.excision_spheres().at("ExcisionSphereB");

  const std::array<double, 4> wt_radius_options{{0.1, 5., 1e-2, 2.3}};
  const std::array<double, 4> bh_radius_options{{0.1, 5., 1e-2, 0.123}};

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {
          wt_excision_sphere,
          bco_domain_creator->create_domain(),
          wt_radius_options,
          bh_radius_options,
      },
      bco_domain_creator->functions_of_time()};

  const double current_time = 1.;
  const Time next_time{{current_time, 2.}, {1, 2}};
  const TimeStepId next_time_step_id{true, 123, next_time};
  std::array<tnsr::I<double, Dim, Frame::Inertial>, 2> particle_pos_vel{};
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> pos_dist(-10., 10.);
  std::uniform_real_distribution<double> vel_dist(-0.2, 0.2);
  particle_pos_vel.at(0) =
      tnsr::I<double, Dim, Frame::Inertial>{{pos_dist(gen), pos_dist(gen), 0.}};
  particle_pos_vel.at(1) =
      tnsr::I<double, Dim, Frame::Inertial>{{vel_dist(gen), vel_dist(gen), 0.}};
  const double initial_expiration_time = 1e-10;
  ActionTesting::emplace_singleton_component_and_initialize<worldtube_chare>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      {1., particle_pos_vel, next_time_step_id, initial_worldtube_radius,
       initial_expiration_time});
  const auto& global_cache = ActionTesting::cache<worldtube_chare>(runner, 0);

  const auto& fots = get<domain::Tags::FunctionsOfTime>(global_cache);
  CHECK(fots.at("Rotation")->time_bounds()[1] ==
        approx(initial_expiration_time));
  CHECK(fots.at("Expansion")->time_bounds()[1] ==
        approx(initial_expiration_time));
  CHECK(fots.at("SizeA")->time_bounds()[1] == approx(initial_expiration_time));
  CHECK(fots.at("SizeB")->time_bounds()[1] == approx(initial_expiration_time));
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  // UpdateFunctionsOfTime
  CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
      make_not_null(&runner), 0));
  // CheckFunctionsOfTimeAreReady
  CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
      make_not_null(&runner), 0));

  const double new_expiration_time_expected = 1.005;

  CHECK(fots.at("Rotation")->time_bounds()[1] ==
        approx(new_expiration_time_expected));
  CHECK(fots.at("Expansion")->time_bounds()[1] ==
        approx(new_expiration_time_expected));
  CHECK(fots.at("SizeA")->time_bounds()[1] ==
        approx(new_expiration_time_expected));
  CHECK(fots.at("SizeB")->time_bounds()[1] ==
        approx(new_expiration_time_expected));

  const auto& expiration_time_tag =
      ActionTesting::get_databox_tag<worldtube_chare, Tags::ExpirationTime>(
          runner, 0);
  CHECK(expiration_time_tag == approx(new_expiration_time_expected));

  const auto& wt_neighbor_block = domain.blocks()[1];
  // does not include shape map
  const auto& grid_to_inertial_map_particle =
      wt_excision_sphere.moving_mesh_grid_to_inertial_map();
  // includes worldtube shape map
  const auto& grid_to_inertial_map_wt_boundary =
      wt_neighbor_block.moving_mesh_grid_to_inertial_map();
  const auto mapped_particle =
      grid_to_inertial_map_particle.coords_frame_velocity_jacobians(
          wt_excision_sphere.center(), 0.5, fots);
  CHECK_ITERABLE_APPROX(std::get<0>(mapped_particle), particle_pos_vel.at(0));
  CHECK_ITERABLE_APPROX(std::get<3>(mapped_particle), particle_pos_vel.at(1));
  const double orbit_radius = get(magnitude(particle_pos_vel[0]));

  auto wt_boundary_point = wt_excision_sphere.center();
  wt_boundary_point[0] += wt_excision_sphere.radius();
  const auto mapped_wt_boundary_data =
      grid_to_inertial_map_wt_boundary.coords_frame_velocity_jacobians(
          wt_boundary_point, 0.5, fots);
  const auto& mapped_wt_boundary_point = std::get<0>(mapped_wt_boundary_data);
  const double wt_radius = get(magnitude(tenex::evaluate<ti::I>(
      mapped_wt_boundary_point(ti::I) - particle_pos_vel.at(0)(ti::I))));
  const double wt_radius_expected = smooth_broken_power_law(
      orbit_radius, wt_radius_options[0], wt_radius_options[1],
      wt_radius_options[2], wt_radius_options[3]);
  CHECK(wt_radius == approx(wt_radius_expected));

  const auto& wt_radius_tag =
      ActionTesting::get_databox_tag<worldtube_chare, Tags::WorldtubeRadius>(
          runner, 0);
  CHECK(wt_radius_tag == approx(wt_radius_expected));

  const auto& bh_neighbor_block = domain.blocks()[12];
  // includes bh shape map
  const auto& grid_to_inertial_map_bh_boundary =
      bh_neighbor_block.moving_mesh_grid_to_inertial_map();
  auto bh_boundary_point = bh_excision_sphere.center();
  bh_boundary_point[0] += bh_excision_sphere.radius();
  const auto mapped_bh_boundary_point = std::get<0>(
      grid_to_inertial_map_bh_boundary.coords_frame_velocity_jacobians(
          bh_boundary_point, 0.5, fots));
  const double bh_radius = get(magnitude(mapped_bh_boundary_point));
  const double bh_radius_expected = smooth_broken_power_law(
      orbit_radius, bh_radius_options[0], bh_radius_options[1],
      bh_radius_options[2], bh_radius_options[3]);
  CHECK(bh_radius == approx(bh_radius_expected));
}

SPECTRE_TEST_CASE("Unit.CurvedScalarWave.Worldtube.UpdateFunctionsOfTime",
                  "[Unit]") {
  test_particle_motion();
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
