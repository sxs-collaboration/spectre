// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;
namespace intrp::Tags {
template <size_t Dim>
struct NumberOfElements;
}  // namespace intrp::Tags

namespace {
struct MockTargetTag
    : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  using temporal_id = ::Tags::TimeStepId;
  using vars_to_interpolate_to_target = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using compute_items_on_target = tmpl::list<>;
  using compute_target_points =
      ::ah::TargetPoints::ApparentHorizon<MockTargetTag, ::Frame::Inertial>;
  using post_interpolation_callbacks = tmpl::list<>;
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<::intrp::Actions::InitializeInterpolator<
              metavariables::volume_dim,
              intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::TimeStepId>,
              intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<Parallel::Phase::Register, tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolator<
          metavariables::volume_dim,
          intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::TimeStepId>,
          intrp::Tags::InterpolatedVarsHolders<Metavariables>>::
          return_tag_list>;
  using component_being_mocked = intrp::Interpolator<Metavariables>;
};

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>,
      Parallel::PhaseActions<
          Parallel::Phase::Register,
          tmpl::list<intrp::Actions::RegisterElementWithInterpolator>>>;
  using initial_databox = db::compute_databox_type<tmpl::list<>>;
};

struct MockMetavariables {
  struct InterpolatorTargetA {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
  };
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolatorTargetA>;

  using component_list = tmpl::list<
      InterpTargetTestHelpers::mock_interpolation_target<MockMetavariables,
                                                         MockTargetTag>,
      mock_interpolator<MockMetavariables>, mock_element<MockMetavariables>>;
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.RegisterElement",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  using metavars = MockMetavariables;
  constexpr size_t Dim = metavars::volume_dim;
  using interp_component = mock_interpolator<metavars>;
  using elem_component = mock_element<metavars>;
  const auto domain_creator = domain::creators::Sphere(
      1.8, 2.2, domain::creators::Sphere::Excision{}, 1_st, 5_st, false);
  const auto block_names = domain_creator.block_names();
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {std::unordered_map<std::string, std::unordered_set<std::string>>{
           {"MockTargetTag", {block_names.begin(), block_names.end()}}},
       ah::OptionHolders::ApparentHorizon<Frame::Inertial>{},
       domain_creator.create_domain(), ::Verbosity::Silent}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_group_component<interp_component>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  }
  for (size_t i = 0; i < 3; i++) {
    ActionTesting::emplace_component<elem_component>(&runner,
                                                     ElementId<Dim>{i});
  }
  const ElementId<Dim> id_0{0};
  const ElementId<Dim> id_1{1};
  const ElementId<Dim> id_2{2};
  // There is no next_action on elem_component, so we don't call it here.
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Register);

  const auto& number_of_elements =
      ActionTesting::get_databox_tag<interp_component,
                                     ::intrp::Tags::NumberOfElements<Dim>>(
          runner, 0);

  const std::string target_name = pretty_type::name<MockTargetTag>();
  REQUIRE(number_of_elements.contains(target_name));

  runner.simple_action<interp_component, ::intrp::Actions::RegisterElement>(
      0, id_0);

  CHECK(number_of_elements.contains(target_name));
  CHECK(number_of_elements.at(target_name).size() == 1);
  CHECK(number_of_elements.at(target_name).contains(id_0));

  runner.simple_action<interp_component, ::intrp::Actions::RegisterElement>(
      0, id_1);

  CHECK(number_of_elements.contains(target_name));
  CHECK(number_of_elements.at(target_name).size() == 2);
  CHECK(number_of_elements.at(target_name).contains(id_1));

  // Call RegisterElementWithInterpolator from element, check if
  // it gets registered.
  ActionTesting::next_action<elem_component>(make_not_null(&runner), id_2);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  runner.invoke_queued_simple_action<interp_component>(0);

  CHECK(number_of_elements.contains(target_name));
  CHECK(number_of_elements.at(target_name).size() == 3);
  CHECK(number_of_elements.at(target_name).contains(id_2));

  // No more queued simple actions.
  CHECK(runner.is_simple_action_queue_empty<interp_component>(0));
  CHECK(runner.is_simple_action_queue_empty<elem_component>(id_0));

  {
    INFO("Deregistration");
    intrp::Actions::RegisterElementWithInterpolator::
        template perform_deregistration<elem_component>(
            ActionTesting::get_databox<elem_component>(make_not_null(&runner),
                                                       id_0),
            ActionTesting::cache<elem_component>(runner, id_0), id_0);
    ActionTesting::invoke_queued_simple_action<interp_component>(
        make_not_null(&runner), 0);
    // No more queued simple actions.
    CHECK(runner.is_simple_action_queue_empty<interp_component>(0));
    CHECK(runner.is_simple_action_queue_empty<elem_component>(id_0));

    CHECK(number_of_elements.contains(target_name));
    CHECK(number_of_elements.at(target_name).size() == 2);
    CHECK_FALSE(number_of_elements.at(target_name).contains(id_0));
    runner.simple_action<interp_component, ::intrp::Actions::DeregisterElement>(
        0, id_1);
    CHECK(number_of_elements.contains(target_name));
    CHECK(number_of_elements.at(target_name).size() == 1);
    CHECK_FALSE(number_of_elements.at(target_name).contains(id_1));
    runner.simple_action<interp_component, ::intrp::Actions::DeregisterElement>(
        0, id_2);
    CHECK(number_of_elements.contains(target_name));
    CHECK(number_of_elements.at(target_name).empty());
  }
}

}  // namespace
