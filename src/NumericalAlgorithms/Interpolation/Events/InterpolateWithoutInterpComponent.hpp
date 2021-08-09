// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/InterpolationTargetVarsFromElement.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Interpolation/PointInfoTag.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
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
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Events {
/// \cond
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename Metavariables, typename Tensors>
class InterpolateWithoutInterpComponent;
/// \endcond

/// Does an interpolation onto an InterpolationTargetTag by calling Actions on
/// the InterpolationTarget component.
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename Metavariables, typename... Tensors>
class InterpolateWithoutInterpComponent<VolumeDim, InterpolationTargetTag,
                                        Metavariables, tmpl::list<Tensors...>>
    : public Event {
  /// \cond
  explicit InterpolateWithoutInterpComponent(
      CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(InterpolateWithoutInterpComponent);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Does interpolation using the given InterpolationTargetTag, "
      "without an Interpolator ParallelComponent.";

  static std::string name() noexcept {
    return Options::name<InterpolationTargetTag>();
  }

  InterpolateWithoutInterpComponent() = default;

  using argument_tags = tmpl::list<typename InterpolationTargetTag::temporal_id,
                                   Tags::InterpPointInfo<Metavariables>,
                                   domain::Tags::Mesh<VolumeDim>, Tensors...>;

  template <typename ParallelComponent>
  void operator()(
      const typename InterpolationTargetTag::temporal_id::type& time_id,
      const typename Tags::InterpPointInfo<Metavariables>::type& point_infos,
      const Mesh<VolumeDim>& mesh, const typename Tensors::type&... tensors,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const /*meta*/) const noexcept {
    // Get element logical coordinates of the target points.
    const auto& block_logical_coords =
        get<Vars::PointInfoTag<InterpolationTargetTag, VolumeDim>>(point_infos);
    const std::vector<ElementId<VolumeDim>> element_ids{{array_index}};
    const auto element_coord_holders =
        element_logical_coordinates(element_ids, block_logical_coords);

    // There is exactly one element_id in the list of element_ids.
    if (element_coord_holders.count(element_ids[0]) == 0) {
      // There are no target points in this element, so we don't need
      // to do anything.
      return;
    }

    // There are points in this element, so interpolate to them and
    // send the interpolated data to the target.  This is done
    // in several steps:
    const auto& element_coord_holder = element_coord_holders.at(element_ids[0]);

    // 1. Get the list of variables
    Variables<typename InterpolationTargetTag::vars_to_interpolate_to_target>
        interp_vars(mesh.number_of_grid_points());

    // Clang-tidy wants extra braces for `if constexpr`
    if constexpr (std::is_same_v<tmpl::list<>, typename InterpolationTargetTag::
                                                   compute_items_on_source>) {
      // 1.a Copy the tensors directly into the variables; no need to
      // make a DataBox because we have no ComputeItems.
      [[maybe_unused]] const auto copy_to_variables = [&interp_vars](
          const auto tensor_tag_v, const auto& tensor) noexcept {
        using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
        get<tensor_tag>(interp_vars) = tensor;
        return 0;
      };
      expand_pack(copy_to_variables(tmpl::type_<Tensors>{}, tensors)...);
    } else {
      // 1.b Make a DataBox and insert ComputeItems
      const auto box = db::create<
          db::AddSimpleTags<Tensors...>,
          db::AddComputeTags<
              typename InterpolationTargetTag::compute_items_on_source>>(
          tensors...);
      // Copy vars_to_interpolate_to_target from databox to vars
      tmpl::for_each<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>(
          [&box, &interp_vars ](auto tag_v) noexcept {
            using tag = typename decltype(tag_v)::type;
            get<tag>(interp_vars) = db::get<tag>(box);
          });
    }

    // 2. Set up interpolator
    intrp::Irregular<VolumeDim> interpolator(
        mesh, element_coord_holder.element_logical_coords);

    // 3. Interpolate and send interpolated data to target
    auto& receiver_proxy = Parallel::get_parallel_component<
        InterpolationTarget<Metavariables, InterpolationTargetTag>>(cache);
    Parallel::simple_action<
        Actions::InterpolationTargetVarsFromElement<InterpolationTargetTag>>(
        receiver_proxy,
        std::vector<Variables<
            typename InterpolationTargetTag::vars_to_interpolate_to_target>>(
            {interpolator.interpolate(interp_vars)}),
        std::vector<std::vector<size_t>>({element_coord_holder.offsets}),
        time_id);
  }

  bool needs_evolved_variables() const noexcept override { return true; }
};

/// \cond
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename Metavariables, typename... Tensors>
PUP::able::PUP_ID InterpolateWithoutInterpComponent<
    VolumeDim, InterpolationTargetTag, Metavariables,
    tmpl::list<Tensors...>>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Events
}  // namespace intrp
