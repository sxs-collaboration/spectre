// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "Domain/Structure/ElementId.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/AddTemporalIdsToInterpolationTarget.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace Tags {
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
}  // namespace domain
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
template <size_t Dim>
class Mesh;
template <size_t VolumeDim>
class ElementId;
namespace intrp {
template <typename Metavariables, typename Tag>
struct InterpolationTarget;
template <typename Metavariables>
struct Interpolator;
namespace Actions {
template <typename TemporalId>
struct InterpolatorReceiveVolumeData;
}  // namespace Actions
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Events {
/// Does an interpolation onto InterpolationTargetTag by calling Actions on
/// the Interpolator and InterpolationTarget components.
template <size_t VolumeDim, typename InterpolationTargetTag, typename Tensors>
class Interpolate;

template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... Tensors>
class Interpolate<VolumeDim, InterpolationTargetTag, tmpl::list<Tensors...>>
    : public Event {
  /// \cond
  explicit Interpolate(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Interpolate);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Starts interpolation onto the given InterpolationTargetTag.";

  static std::string name() noexcept {
    return Options::name<InterpolationTargetTag>();
  }

  Interpolate() = default;

  using argument_tags = tmpl::list<typename InterpolationTargetTag::temporal_id,
                                   domain::Tags::Mesh<VolumeDim>, Tensors...>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const typename InterpolationTargetTag::temporal_id::type& temporal_id,
      const Mesh<VolumeDim>& mesh, const typename Tensors::type&... tensors,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const /*meta*/) const noexcept {
    Variables<tmpl::list<Tensors...>> interp_vars(mesh.number_of_grid_points());
    const auto copy_to_variables = [&interp_vars](const auto tensor_tag_v,
                                                  const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      get<tensor_tag>(interp_vars) = tensor;
      return 0;
    };
    (void) copy_to_variables; // GCC warns unused variable if Tensors is empty.
    expand_pack(copy_to_variables(tmpl::type_<Tensors>{}, tensors)...);

    // Send volume data to the Interpolator, to trigger interpolation.
    auto& interpolator =
        *::Parallel::get_parallel_component<Interpolator<Metavariables>>(cache)
             .ckLocalBranch();
    Parallel::simple_action<Actions::InterpolatorReceiveVolumeData<
        typename InterpolationTargetTag::temporal_id>>(
        interpolator, temporal_id, ElementId<VolumeDim>(array_index), mesh,
        interp_vars);

    // Tell the interpolation target that it should interpolate.
    auto& target = Parallel::get_parallel_component<
        InterpolationTarget<Metavariables, InterpolationTargetTag>>(cache);
    Parallel::simple_action<
        Actions::AddTemporalIdsToInterpolationTarget<InterpolationTargetTag>>(
        target, std::vector<typename InterpolationTargetTag::temporal_id::type>{
                    temporal_id});
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const noexcept {
    return true;
  }

  bool needs_evolved_variables() const noexcept override { return true; }
};

/// \cond
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... Tensors>
PUP::able::PUP_ID Interpolate<VolumeDim, InterpolationTargetTag,
                              tmpl::list<Tensors...>>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Events
}  // namespace intrp
