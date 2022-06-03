// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "ApparentHorizons/ObjectLabel.hpp"
#include "ControlSystem/ApparentHorizons/Measurements.hpp"
#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Expansion.hpp"
#include "ControlSystem/ControlErrors/Rotation.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/DataVectorHelpers.hpp"
#include "ControlSystem/Systems/Expansion.hpp"
#include "ControlSystem/Systems/Rotation.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "ControlSystem/UpdateControlSystem.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestingFramework.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace OptionTags {
struct InitialTime;
}  // namespace OptionTags
/// \endcond

namespace control_system::TestHelpers {
template <typename ControlSystem>
using init_simple_tags =
    tmpl::list<control_system::Tags::Averager<ControlSystem>,
               control_system::Tags::TimescaleTuner<ControlSystem>,
               control_system::Tags::Controller<ControlSystem>,
               control_system::Tags::ControlError<ControlSystem>,
               control_system::Tags::WriteDataToDisk,
               control_system::Tags::IsActive<ControlSystem>,
               typename ControlSystem::MeasurementQueue>;

template <typename Metavariables, typename ControlSystem>
struct MockControlComponent {
  using array_index = int;
  using component_being_mocked = ControlComponent<Metavariables, ControlSystem>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;

  using system = ControlSystem;

  using simple_tags = init_simple_tags<ControlSystem>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavars>
struct MockElementComponent {
  using array_index = int;
  using chare_type = ActionTesting::MockArrayChare;

  using metavariables = Metavars;

  using simple_tags = tmpl::list<>;

  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<Metavars::volume_dim>>;

  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize,
                 control_system::Tags::MeasurementTimescales>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename metavariables::Phase,
                                        metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavars>
struct MockObserverWriter {
  using component_being_mocked = observers::ObserverWriter<Metavars>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using const_global_cache_tags = tmpl::list<>;

  using initialization_tags = tmpl::list<>;

  using metavariables = Metavars;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavars::Phase, Metavars::Phase::Initialization, tmpl::list<>>>;
};

template <size_t RotationDerivOrder, size_t ExpansionDerivOrder>
struct MockMetavars {
  static constexpr size_t volume_dim = 3;

  enum class Phase { Initialization, Testing, Exit };

  using metavars = MockMetavars<RotationDerivOrder, ExpansionDerivOrder>;

  using observed_reduction_data_tags = tmpl::list<>;

  static constexpr bool using_expansion = ExpansionDerivOrder != 0;
  static constexpr bool using_rotation = RotationDerivOrder != 0;

  // Even if we aren't using certain control systems, we still need valid deriv
  // orders becuse everything is constructed by default in the SystemHelper. The
  // bool above just determines if the functions of time are actually created or
  // not because that's what matters
  static constexpr size_t exp_deriv_order =
      using_expansion ? ExpansionDerivOrder : 2;
  static constexpr size_t rot_deriv_order =
      using_rotation ? RotationDerivOrder : 2;

  using element_component = MockElementComponent<metavars>;

  using expansion_system = control_system::Systems::Expansion<exp_deriv_order>;
  using rotation_system = control_system::Systems::Rotation<rot_deriv_order>;

  using expansion_component = MockControlComponent<metavars, expansion_system>;
  using rotation_component = MockControlComponent<metavars, rotation_system>;

  using observer_component = MockObserverWriter<metavars>;

  using control_components = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<using_expansion, expansion_component, tmpl::list<>>,
      tmpl::conditional_t<using_rotation, rotation_component, tmpl::list<>>>>;

  using component_list = tmpl::flatten<
      tmpl::list<observer_component, element_component, control_components>>;
};

/*!
 * \brief Helper struct for testing basic control systems
 *
 * To signify which control systems you want, set the corresponding
 * DerivOrder. To turn control systems off, put 0 for their DerivOrder in the
 * templates of the metavariables.
 *
 * Ideally we'd construct the runner here and just pass that to the test to
 * simplify as must of the work as possible, but MockRuntimeSystems aren't
 * copy- or move-able so we have to make the necessary info available. The
 * simplist way to do this was to have functions that return references to the
 * member variables.
 *
 * \note Translation control isn't supported yet. It will be added in the
 * future.
 */
template <typename Metavars>
struct SystemHelper {
  static constexpr size_t exp_deriv_order = Metavars::exp_deriv_order;
  static constexpr size_t rot_deriv_order = Metavars::rot_deriv_order;

  static constexpr bool using_expansion = Metavars::using_expansion;
  static constexpr bool using_rotation = Metavars::using_rotation;

  using expansion_system = typename Metavars::expansion_system;
  using rotation_system = typename Metavars::rotation_system;

  using element_component = typename Metavars::element_component;
  using control_components = typename Metavars::control_components;

  using expansion_init_simple_tags = init_simple_tags<expansion_system>;
  using rotation_init_simple_tags = init_simple_tags<rotation_system>;

  // Members that may be moved out of this struct once they are
  // constructed
  auto& domain() { return domain_; }
  auto& initial_functions_of_time() { return initial_functions_of_time_; }
  auto& initial_measurement_timescales() {
    return initial_measurement_timescales_;
  }

  // Members that won't be moved out of this struct
  const auto& init_exp_tuple() { return init_exp_tuple_; }
  const auto& init_rot_tuple() { return init_rot_tuple_; }
  const auto& grid_position_of_a() { return grid_position_of_a_; }
  const auto& grid_position_of_b() { return grid_position_of_b_; }
  const auto& expansion_name() { return expansion_name_; }
  const auto& rotation_name() { return rotation_name_; }

  void setup_control_system_test(const double initial_time,
                                 const double initial_separation,
                                 const std::string& option_string) {
    initial_time_ = initial_time;

    // We don't need a real domain, just one that has the correct excision
    // sphere centers because the control errors use the `excision_spheres()`
    // member of a domain to get the centers. The names are chosen to match the
    // BinaryCompactObject domain, which the control errors were based on and
    // have these specific names hard-coded into them.
    domain_ =
        Domain<3>{{},
                  {},
                  {{"ObjectAExcisionSphere",
                    ExcisionSphere<3>{1.0,
                                      {{-0.5 * initial_separation, 0.0, 0.0}},
                                      {{0, Direction<3>::lower_zeta()},
                                       {1, Direction<3>::lower_zeta()},
                                       {2, Direction<3>::lower_zeta()},
                                       {3, Direction<3>::lower_zeta()},
                                       {4, Direction<3>::lower_zeta()},
                                       {5, Direction<3>::lower_zeta()}}}},
                   {"ObjectBExcisionSphere",
                    ExcisionSphere<3>{1.0,
                                      {{+0.5 * initial_separation, 0.0, 0.0}},
                                      {{0, Direction<3>::lower_zeta()},
                                       {1, Direction<3>::lower_zeta()},
                                       {2, Direction<3>::lower_zeta()},
                                       {3, Direction<3>::lower_zeta()},
                                       {4, Direction<3>::lower_zeta()},
                                       {5, Direction<3>::lower_zeta()}}}}}};

    // Initial parameters needed. Expiration times would normally be set during
    // option parsing, and measurement timescales during initialization so we
    // have to do them manually here instead.
    if constexpr (using_expansion) {
      init_exp_tuple_ = parse_options<expansion_system>(option_string);
      auto& exp_averager =
          get<control_system::Tags::Averager<expansion_system>>(
              init_exp_tuple_);
      const auto& exp_controller =
          get<control_system::Tags::Controller<expansion_system>>(
              init_exp_tuple_);
      const auto& exp_tuner =
          get<control_system::Tags::TimescaleTuner<expansion_system>>(
              init_exp_tuple_);

      const std::array<DataVector, 1> expansion_measurement_timescale{
          {control_system::calculate_measurement_timescales(exp_controller,
                                                            exp_tuner)}};
      exp_averager.assign_time_between_measurements(
          min(expansion_measurement_timescale[0]));

      const double initial_expansion_expiration_time =
          exp_controller.get_update_fraction() *
          min(exp_tuner.current_timescale());
      const double initial_expansion = 1.0;
      const double expansion_velocity_outer_boundary = 0.0;
      const double decay_timescale_outer_boundary = 0.05;
      auto init_func_expansion =
          make_array<exp_deriv_order + 1, DataVector>(DataVector{1, 0.0});
      init_func_expansion[0][0] = initial_expansion;

      initial_functions_of_time_[expansion_name_] = std::make_unique<
          domain::FunctionsOfTime::PiecewisePolynomial<exp_deriv_order>>(
          initial_time_, init_func_expansion,
          initial_expansion_expiration_time);
      initial_functions_of_time_[expansion_name_ + "OuterBoundary"s] =
          std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
              initial_expansion, initial_time_,
              expansion_velocity_outer_boundary,
              decay_timescale_outer_boundary);
      initial_measurement_timescales_[expansion_name_] =
          std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
              initial_time_, expansion_measurement_timescale,
              initial_expansion_expiration_time);
    }
    if constexpr (using_rotation) {
      init_rot_tuple_ = parse_options<rotation_system>(option_string);
      auto& rot_averager =
          get<control_system::Tags::Averager<rotation_system>>(init_rot_tuple_);
      const auto& rot_controller =
          get<control_system::Tags::Controller<rotation_system>>(
              init_rot_tuple_);
      const auto& rot_tuner =
          get<control_system::Tags::TimescaleTuner<rotation_system>>(
              init_rot_tuple_);

      const std::array<DataVector, 1> rotation_measurement_timescale{
          {control_system::calculate_measurement_timescales(rot_controller,
                                                            rot_tuner)}};
      rot_averager.assign_time_between_measurements(
          min(rotation_measurement_timescale[0]));

      const double initial_rotation_rotation_time =
          rot_controller.get_update_fraction() *
          min(rot_tuner.current_timescale());
      const double initial_omega_z = 0.01;
      auto init_func_rotation =
          make_array<rot_deriv_order + 1, DataVector>(DataVector{3, 0.0});
      init_func_rotation[1][2] = initial_omega_z;
      auto init_quaternion = make_array<1, DataVector>(DataVector{4, 0.0});
      init_quaternion[0][0] = 1.0;

      initial_functions_of_time_[rotation_name_] = std::make_unique<
          domain::FunctionsOfTime::QuaternionFunctionOfTime<rot_deriv_order>>(
          initial_time_, init_quaternion, init_func_rotation,
          initial_rotation_rotation_time);
      initial_measurement_timescales_[rotation_name_] =
          std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
              initial_time_, rotation_measurement_timescale,
              initial_rotation_rotation_time);
    }
  }

  template <typename Generator, typename F, typename CoordMap>
  void run_control_system_test(
      ActionTesting::MockRuntimeSystem<Metavars>& runner,
      const double final_time, gsl::not_null<Generator*> generator,
      const F position_function, const CoordMap& coord_map) {
    // Allocate now because we need these variables outside the loop
    std::pair<std::array<double, 3>, std::array<double, 3>> positions{};
    double time = initial_time_;
    std::optional<double> prev_time{};
    double dt = 0.0;

    auto& cache = ActionTesting::cache<element_component>(runner, 0);
    const auto& functions_of_time =
        Parallel::get<domain::Tags::FunctionsOfTime>(cache);
    const auto& measurement_timescales =
        Parallel::get<control_system::Tags::MeasurementTimescales>(cache);

    // Start loop
    while (time < final_time) {
      // Setup the measurement Id. This would normally be created in the control
      // system event.
      const LinkedMessageId<double> measurement_id{time, prev_time};

      // This whole switching between tensors and arrays is annoying and
      // clunky, but it's the best that could be done at the moment without
      // changing BinaryTrajectories to return tensors, which doesn't seem like
      // a good idea.

      // Get trajectory in "inertial coordinates" as arrays
      positions = position_function(time);

      // Covert arrays to tensor so we can pass them into the coordinate map
      const tnsr::I<double, 3, Frame::Inertial> inertial_position_of_a(
          positions.first);
      const tnsr::I<double, 3, Frame::Inertial> inertial_position_of_b(
          positions.second);

      // Convert to "grid coordinates"
      const auto grid_position_of_a_tnsr =
          *coord_map.inverse(inertial_position_of_a, time, functions_of_time);
      const auto grid_position_of_b_tnsr =
          *coord_map.inverse(inertial_position_of_b, time, functions_of_time);

      // Convert tensors back to arrays so we can pass them to the control
      // systems
      for (size_t i = 0; i < 3; i++) {
        gsl::at(grid_position_of_a_, i) = grid_position_of_a_tnsr.get(i);
        gsl::at(grid_position_of_b_, i) = grid_position_of_b_tnsr.get(i);
      }

      // Construct strahlkorpers to pass to control systems. Only the centers
      // matter.
      const Strahlkorper<Frame::Grid> horizon_a{2, 2, 1.0, grid_position_of_a_};
      const Strahlkorper<Frame::Grid> horizon_b{2, 2, 1.0, grid_position_of_b_};

      // Apply measurements
      tmpl::for_each<control_components>([&runner, &generator, &measurement_id,
                                          &horizon_a, &cache,
                                          &horizon_b](auto control_component) {
        using component = tmpl::type_from<decltype(control_component)>;
        using system = typename component::system;
        system::process_measurement::apply(
            ah::BothHorizons::FindHorizon<::ah::ObjectLabel::A>{}, horizon_a,
            cache, measurement_id);
        CHECK(ActionTesting::number_of_queued_simple_actions<component>(
                  runner, 0) == 1);
        system::process_measurement::apply(
            ah::BothHorizons::FindHorizon<::ah::ObjectLabel::B>{}, horizon_b,
            cache, measurement_id);
        CHECK(ActionTesting::number_of_queued_simple_actions<component>(
                  runner, 0) == 2);
        // We invoke a random measurement because during a normal simulation
        // we don't know which measurement will reach the control system
        // first because of charm++ communication
        ActionTesting::invoke_random_queued_simple_action<control_components>(
            make_not_null(&runner), generator,
            ActionTesting::array_indices_with_queued_simple_actions<
                control_components>(make_not_null(&runner)));
        ActionTesting::invoke_queued_simple_action<component>(
            make_not_null(&runner), 0);
      });

      // At this point, the control systems for each transformation should have
      // done their thing and updated the functions of time (if they had enough
      // data).

      // Our dt is set by the smallest measurement timescale. The control system
      // updates these timescales when it updates the functions of time
      prev_time = time;
      dt = std::numeric_limits<double>::max();
      for (auto& [name, measurement_timescale] : measurement_timescales) {
        // Avoid compiler warning with gcc-7
        (void)name;
        dt = std::min(dt, min(measurement_timescale->func(time)[0]));
      }
      time += dt;
    }

    // Get analytic position in inertial coordinates
    positions = position_function(final_time);

    // Get position of objects in grid coordinates using the coordinate map that
    // has had its functions of time updated by the control system
    const tnsr::I<double, 3, Frame::Inertial> inertial_position_of_a(
        positions.first);
    const tnsr::I<double, 3, Frame::Inertial> inertial_position_of_b(
        positions.second);
    const auto grid_position_of_a_tnsr = *coord_map.inverse(
        inertial_position_of_a, final_time, functions_of_time);
    const auto grid_position_of_b_tnsr = *coord_map.inverse(
        inertial_position_of_b, final_time, functions_of_time);
    for (size_t i = 0; i < 3; i++) {
      gsl::at(grid_position_of_a_, i) = grid_position_of_a_tnsr.get(i);
      gsl::at(grid_position_of_b_, i) = grid_position_of_b_tnsr.get(i);
    }
  }

 private:
  template <typename Component>
  using option_tag = control_system::OptionTags::ControlSystemInputs<
      typename Component::system>;
  using option_list = tmpl::push_back<
      tmpl::remove_duplicates<tmpl::transform<
          control_components, tmpl::bind<option_tag, tmpl::_1>>>,
      control_system::OptionTags::WriteDataToDisk, ::OptionTags::InitialTime>;
  template <typename System>
  using creatable_tags =
      tmpl::list_difference<init_simple_tags<System>,
                            tmpl::list<typename System::MeasurementQueue>>;

  template <typename System>
  tuples::tagged_tuple_from_typelist<init_simple_tags<System>> parse_options(
      const std::string& option_string) {
    Options::Parser<option_list> parser{"Peter Parker the option parser."};
    parser.parse(option_string);
    const tuples::tagged_tuple_from_typelist<option_list> options =
        parser.template apply<option_list, Metavars>([](auto... args) {
          return tuples::tagged_tuple_from_typelist<option_list>(
              std::move(args)...);
        });

    tuples::tagged_tuple_from_typelist<creatable_tags<System>> created_tags =
        Parallel::create_from_options<Metavars>(options,
                                                creatable_tags<System>{});

    return tuples::tagged_tuple_from_typelist<init_simple_tags<System>>{
        get<control_system::Tags::Averager<System>>(created_tags),
        get<control_system::Tags::TimescaleTuner<System>>(created_tags),
        get<control_system::Tags::Controller<System>>(created_tags),
        get<control_system::Tags::ControlError<System>>(created_tags),
        get<control_system::Tags::WriteDataToDisk>(created_tags), true,
        // Just need an empty queue. It will get filled in as the control
        // system is updated
        LinkedMessageQueue<
            double, tmpl::list<QueueTags::Center<::ah::ObjectLabel::A>,
                               QueueTags::Center<::ah::ObjectLabel::B>>>{}};
  }

  // Members that may be moved out of this struct once they are
  // constructed
  Domain<3> domain_;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      initial_functions_of_time_{};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      initial_measurement_timescales_{};

  // Members that won't be moved out of this struct
  tuples::tagged_tuple_from_typelist<expansion_init_simple_tags>
      init_exp_tuple_;
  tuples::tagged_tuple_from_typelist<rotation_init_simple_tags> init_rot_tuple_;
  std::array<double, 3> grid_position_of_a_{};
  std::array<double, 3> grid_position_of_b_{};
  const std::string expansion_name_{expansion_system::name()};
  const std::string rotation_name_{rotation_system::name()};
  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
};
}  // namespace control_system::TestHelpers
