// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
// IWYU pragma: no_forward_declare Tensor
namespace intrp {
namespace Actions {
template <typename InterpolationTargetTag>
struct AddTemporalIdsToInterpolationTarget;
template <typename InterpolationTargetTag>
struct InterpolationTargetReceiveVars;
}  // namespace Actions
}  // namespace intrp
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
namespace Tags {
template <typename Metavariables>
struct InterpolatedVarsHolders;
template <typename Metavariables>
struct TemporalIds;
template <typename Metavariables>
struct VolumeVarsInfo;
}  // namespace Tags
}  // namespace intrp
/// \endcond

namespace {

// Simple DataBoxItems for test.
namespace Tags {
struct Square : db::SimpleTag {
  static std::string name() noexcept { return "Square"; }
  using type = Scalar<DataVector>;
};
struct SquareComputeItem : Square, db::ComputeTag {
  static std::string name() noexcept { return "Square"; }
  static Scalar<DataVector> function(const Scalar<DataVector>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.0);
    get<>(result) = square(get<>(x));
    return result;
  }
  using argument_tags = tmpl::list<gr::Tags::Lapse<DataVector>>;
};
}  // namespace Tags

template <typename InterpolationTargetTag>
struct MockInterpolationTargetReceiveVars {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
          DbTags, typename intrp::Tags::TemporalIds<Metavariables>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const std::vector<db::item_type<::Tags::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>>&
          vars_src,
      const std::vector<std::vector<size_t>>& global_offsets) noexcept {
    size_t number_of_interpolated_points = 0;
    for (size_t i = 0; i < global_offsets.size(); ++i) {
      Scalar<DataVector> expected_vars{global_offsets[i].size()};
      number_of_interpolated_points += global_offsets[i].size();
      for (size_t s = 0; s < global_offsets[i].size(); ++s) {
        // Coords at this point. They are the same as the input coordinates,
        // but in strange order because of global_offsets.
        std::array<double, Metavariables::domain_dim> coords{
            {1.0 + 0.1 * global_offsets[i][s],
             1.0 + 0.12 * global_offsets[i][s],
             1.0 + 0.14 * global_offsets[i][s]}};
        const double lapse =
            2.0 * get<0>(coords) + 3.0 * get<1>(coords) +
            5.0 * get<2>(coords);  // Same formula as input lapse.
        get<>(expected_vars)[s] = square(lapse);
      }
      // We don't have that many points, so interpolation is good for
      // only a few digits.
      Approx custom_approx = Approx::custom().epsilon(1.e-5).scale(1.0);
      CHECK_ITERABLE_CUSTOM_APPROX(
          expected_vars, get<Tags::Square>(vars_src[i]), custom_approx);
    }
    // Make sure we have interpolated at the correct number of points.
    CHECK(number_of_interpolated_points == 15);
    // Change something in the DataBox so we can test that this function was
    // indeed called.  Put some unusual temporal_id into Tags::TemporalIds.
    // This is not the usual usage of Tags::TemporalIds; this is done just
    // for the test.
    Slab slab(0.0, 1.0);
    TimeId temporal_id(true, 0, Time(slab, Rational(111, 135)));
    db::mutate<intrp::Tags::TemporalIds<Metavariables>>(
        make_not_null(&box), [&temporal_id](
                                 const gsl::not_null<db::item_type<
                                     intrp::Tags::TemporalIds<Metavariables>>*>
                                     temporal_ids) noexcept {
          temporal_ids->push_back(temporal_id);
        });
  }
};

size_t called_mock_add_temporal_ids_to_interpolation_target = 0;
template <typename InterpolationTargetTag>
struct MockAddTemporalIdsToInterpolationTarget {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<typename Metavariables::temporal_id::type>&&
                    /*temporal_ids*/) noexcept {
    // We are not testing this Action here.
    // Do nothing except make sure it is called once.
    ++called_mock_add_temporal_ids_to_interpolation_target;
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables>>;
  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::InterpolationTargetReceiveVars<
                     typename Metavariables::InterpolationTargetA>,
                 intrp::Actions::AddTemporalIdsToInterpolationTarget<
                     typename Metavariables::InterpolationTargetA>>;
  using with_these_simple_actions =
      tmpl::list<MockInterpolationTargetReceiveVars<
                     typename Metavariables::InterpolationTargetA>,
                 MockAddTemporalIdsToInterpolationTarget<
                     typename Metavariables::InterpolationTargetA>>;
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<typename intrp::Actions::InitializeInterpolator::
                                   template return_tag_list<Metavariables>>;
  using component_being_mocked = void;  // not needed.
};

struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target = tmpl::list<Tags::Square>;
    using compute_items_on_source = tmpl::list<Tags::SquareComputeItem>;
  };
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  using temporal_id = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.ReceiveVolumeData",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTagTarget =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolation_target<metavars, metavars::InterpolationTargetA>>;
  using MockDistributedObjectsTagInterpolator =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolator<metavars>>;

  tuples::get<MockDistributedObjectsTagTarget>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<mock_interpolation_target<
                   metavars, metavars::InterpolationTargetA>>{});

  // Make an InterpolatedVarsHolders containing the target points.
  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{7, 7}}, false);
  const auto domain = domain_creator.create_domain();
  Slab slab(0.0, 1.0);
  TimeId temporal_id(true, 0, Time(slab, Rational(11, 15)));
  auto vars_holders = [&domain, &temporal_id]() {
    const size_t n_pts = 15;
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_pts);
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < n_pts; ++i) {
        points.get(d)[i] = 1.0 + (0.1 + 0.02 * d) * i;  // Chosen by hand.
      }
    }
    auto coords = block_logical_coordinates(domain, points);
    db::item_type<intrp::Tags::InterpolatedVarsHolders<metavars>>
        vars_holders_l{};
    auto& vars_infos =
        get<intrp::Vars::HolderTag<metavars::InterpolationTargetA, metavars>>(
            vars_holders_l)
            .infos;
    vars_infos.emplace(std::make_pair(
        temporal_id,
        intrp::Vars::Info<3, typename metavars::InterpolationTargetA::
                                 vars_to_interpolate_to_target>{
            std::move(coords)}));
    return vars_holders_l;
  }();

  // Set initial DataBox of Interpolator to contain an InterpolatedVarsHolders
  // containing the target points.
  tuples::get<MockDistributedObjectsTagInterpolator>(dist_objects)
      .emplace(
          0,
          ActionTesting::MockDistributedObject<mock_interpolator<metavars>>{
              db::create<db::get_items<intrp::Actions::InitializeInterpolator::
                                           return_tag_list<metavars>>>(
                  0_st, db::item_type<intrp::Tags::VolumeVarsInfo<metavars>>{},
                  db::item_type<intrp::Tags::InterpolatedVarsHolders<metavars>>{
                      vars_holders})});

  MockRuntimeSystem runner{{}, std::move(dist_objects)};

  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      ::intrp::Actions::InitializeInterpolationTarget<
          metavars::InterpolationTargetA>>(0, domain_creator.create_domain());

  const auto& box_target =
      runner
          .template algorithms<mock_interpolation_target<
              metavars, metavars::InterpolationTargetA>>()
          .at(0)
          .template get_databox<typename mock_interpolation_target<
              metavars, metavars::InterpolationTargetA>::initial_databox>();

  // Create Element_ids.
  std::vector<ElementId<3>> element_ids{};
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator.initial_refinement_levels()[block.id()];
    auto elem_ids = initial_element_ids(block.id(), initial_ref_levs);
    element_ids.insert(element_ids.end(), elem_ids.begin(), elem_ids.end());
  }

  // Tell the interpolator how many elements there are by registering
  // each one.
  for (size_t i = 0; i < element_ids.size(); ++i) {
    runner.simple_action<mock_interpolator<metavars>,
                         intrp::Actions::RegisterElement>(0);
  }

  // Create volume data and send it to the interpolator.
  for (const auto& element_id : element_ids) {
    const auto& block = domain.blocks()[element_id.block_id()];
    ::Mesh<3> mesh{domain_creator.initial_extents()[element_id.block_id()],
                   Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    ElementMap<3, Frame::Inertial> map{element_id,
                                       block.coordinate_map().get_clone()};
    const auto inertial_coords = map(logical_coordinates(mesh));
    db::item_type<
        ::Tags::Variables<typename metavars::interpolator_source_vars>>
        output_vars(mesh.number_of_grid_points());
    auto& lapse = get<gr::Tags::Lapse<DataVector>>(output_vars);

    // Fill lapse with some analytic solution.
    get<>(lapse) = 2.0 * get<0>(inertial_coords) +
                   3.0 * get<1>(inertial_coords) +
                   5.0 * get<2>(inertial_coords);

    // Call the action on each element_id.
    runner.simple_action<mock_interpolator<metavars>,
                         ::intrp::Actions::InterpolatorReceiveVolumeData>(
        0, temporal_id, element_id, mesh, std::move(output_vars));
  }

  // Should be no temporal_ids in the target box, since we never
  // put any there.
  CHECK(db::get<intrp::Tags::TemporalIds<metavars>>(box_target).empty());

  // Should be two queued simple actions. First is
  // MockAddTemporalIdsToInterpolationTarget.
  runner.invoke_queued_simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>>(0);

  // Make sure MockAddTemporalIdsToInterpolationTarget was called once.
  CHECK(called_mock_add_temporal_ids_to_interpolation_target == 1);

  // Should be one queued simple action, MockInterpolationTargetReceiveVars.
  runner.invoke_queued_simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>>(0);

  // Make sure that MockInterpolationTargetReceiveVars was called,
  // by looking for a funny temporal_id that it inserts for the specific
  // purpose of this test.
  CHECK(db::get<intrp::Tags::TemporalIds<metavars>>(box_target).front() ==
        TimeId(true, 0, Time(Slab(0.0, 1.0), Rational(111, 135))));

  // No more queued simple actions.
  CHECK(runner.is_simple_action_queue_empty<
        mock_interpolation_target<metavars, metavars::InterpolationTargetA>>(
      0));
}
}  // namespace
