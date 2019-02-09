// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp" // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class ElementId;
namespace intrp {
}  // namespace intrp
/// \endcond

namespace {

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolator::
          template return_tag_list<Metavariables>>;
};

struct MockMetavariables {
  struct InterpolationTagA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  struct InterpolationTagB {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  struct InterpolationTagC {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  using temporal_id = ::Tags::TimeId;
  static constexpr size_t domain_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags =
      tmpl::list<InterpolationTagA, InterpolationTagB, InterpolationTagC>;

  using component_list = tmpl::list<mock_interpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.CleanUp", "[Unit]") {
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolator<metavars>>;

  Slab slab(0.0, 1.0);
  TimeId temporal_id(true, 0, Time(slab, Rational(12, 13)));

  // Make a VolumeVarsInfo that contains a single temporal_id but
  // no data (since we don't need data for this test).
  std::unordered_map<
      typename metavars::temporal_id::type,
      std::unordered_map<ElementId<3>,
                         intrp::Tags::VolumeVarsInfo<metavars>::Info>>
      volume_vars_info{{temporal_id, {}}};

  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(
          0, ActionTesting::MockDistributedObject<mock_interpolator<metavars>>{
                 db::create<db::get_items<
                     typename intrp::Actions::InitializeInterpolator::
                         template return_tag_list<metavars>>>(
                     0_st,
                     db::item_type<intrp::Tags::VolumeVarsInfo<metavars>>{
                         std::move(volume_vars_info)},
                     db::item_type<
                         intrp::Tags::InterpolatedVarsHolders<metavars>>{})});

  MockRuntimeSystem runner{{}, std::move(dist_objects)};

  const auto& box =
      runner.template algorithms<mock_interpolator<metavars>>()
          .at(0)
          .template get_databox<
              typename mock_interpolator<metavars>::initial_databox>();

  // There should be one temporal_id in VolumeVarsInfo.
  CHECK(db::get<intrp::Tags::VolumeVarsInfo<metavars>>(box).size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for each tag.
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagB, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.empty());

  // Call the action on InterpolationTagA
  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagA>>(
      0, temporal_id);

  // There should still be one temporal_id in VolumeVarsInfo.
  CHECK(db::get<intrp::Tags::VolumeVarsInfo<metavars>>(box).size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for B and C,
  // but should contain the correct temporal_id for A.
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.count(temporal_id) ==
        1);
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagB, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.empty());

  // Call the action on InterpolationTagC
  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagC>>(
      0, temporal_id);

  // There should still be one temporal_id in VolumeVarsInfo.
  CHECK(db::get<intrp::Tags::VolumeVarsInfo<metavars>>(box).size() == 1);

  // temporal_ids_when_data_has_been_interpolated should be empty for B,
  // but should contain the correct temporal_id for A and C.
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.count(temporal_id) ==
        1);
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.size() == 1);
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.count(temporal_id) ==
        1);
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagB, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.empty());

  // Call the action on InterpolationTagB. This will clean up everything
  // since all the tags have now cleaned up.
  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::CleanUpInterpolator<metavars::InterpolationTagB>>(
      0, temporal_id);

  // There should be no temporal_ids in VolumeVarsInfo.
  CHECK(db::get<intrp::Tags::VolumeVarsInfo<metavars>>(box).empty());

  // temporal_ids_when_data_has_been_interpolated should be empty for each tag.
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagA, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagB, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(get<intrp::Vars::HolderTag<metavars::InterpolationTagC, metavars>>(
            db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box))
            .temporal_ids_when_data_has_been_interpolated.empty());

  // There should be no queued actions; verify this.
  CHECK(runner.is_simple_action_queue_empty<mock_interpolator<metavars>>(0));
}

}  // namespace
