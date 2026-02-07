// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/ArrayCollection/PerformAlgorithmOnElement.hpp"
#include "Parallel/ArrayCollection/Tags/ElementLocations.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/GetItemFromDistributedObject.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/Interpolation/Events/GetComputeItemsOnSource.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
template <size_t VolumeDim>
class ElementId;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct Time;
}  // namespace Tags
namespace Events::Tags {
template <size_t Dim>
struct ObserverMesh;
}  // namespace Events::Tags
/// \endcond

namespace intrp::Events {
/*!
 * \brief Event that combines the `intrp::Events::interpolate` event with
 * `dg::Events::ObserveFields` specifically for common horizon finding.
 *
 * \details This is just a thin wrapper around each event and doesn't do
 * anything extra.
 */
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename InterpolatorSourceVarTags, typename Tensors,
          typename NonTensorComputeTagsList = tmpl::list<>>
class FindCommonHorizon;

template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... InterpolatorSourceVarTags, typename... Tensors,
          typename... NonTensorComputeTags>
class FindCommonHorizon<
    VolumeDim, InterpolationTargetTag, tmpl::list<InterpolatorSourceVarTags...>,
    tmpl::list<Tensors...>, tmpl::list<NonTensorComputeTags...>>
    : public Event {
  using ObserveFieldsEvent =
      dg::Events::ObserveFields<VolumeDim, tmpl::list<Tensors...>,
                                tmpl::list<NonTensorComputeTags...>>;
  using InterpolateEvent =
      intrp::Events::Interpolate<VolumeDim, InterpolationTargetTag,
                                 tmpl::list<InterpolatorSourceVarTags...>>;

 public:
  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FindCommonHorizon);  // NOLINT
  /// \endcond

  using options = tmpl::append<typename ObserveFieldsEvent::options,
                               typename InterpolateEvent::options>;
  static constexpr Options::String help =
      "Starts a common horizon find and sends the evolved variables to be "
      "written to disk.";

  static std::string name() {
    return pretty_type::name<InterpolationTargetTag>();
  }

  FindCommonHorizon() = default;

  FindCommonHorizon(
      const std::string& subfile_name,
      FloatingPointType coordinates_floating_point_type,
      const std::vector<FloatingPointType>& floating_point_types,
      const std::vector<std::string>& variables_to_observe,
      std::optional<std::vector<std::string>> active_block_or_block_groups = {},
      std::optional<Mesh<VolumeDim>> interpolation_mesh = {},
      std::optional<Mesh<VolumeDim>> projection_mesh = {},
      const Options::Context& context = {})
      : observe_fields_event_(subfile_name, coordinates_floating_point_type,
                              floating_point_types, variables_to_observe,
                              active_block_or_block_groups, interpolation_mesh,
                              projection_mesh, std::nullopt, context),
        interpolate_event_(subfile_name) {}

  using compute_tags_for_observation_box = tmpl::remove_duplicates<tmpl::append<
      typename ObserveFieldsEvent::compute_tags_for_observation_box,
      typename InterpolateEvent::compute_tags_for_observation_box>>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::ObservationBox,
                                   ::Events::Tags::ObserverMesh<VolumeDim>>;

  template <typename DataBoxType, typename ComputeTagsList,
            typename Metavariables, typename ParallelComponent>
  void operator()(const ObservationBox<DataBoxType, ComputeTagsList>& box,
                  const Mesh<VolumeDim>& mesh,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<VolumeDim>& element_id,
                  const ParallelComponent* const component,
                  const ObservationValue& observation_value) const {
    observe_fields_event_(box, mesh, cache, element_id, component,
                          observation_value);

    interpolate_event_(get<typename InterpolationTargetTag::temporal_id>(box),
                       mesh, get<InterpolatorSourceVarTags>(box)..., cache,
                       element_id, component, observation_value);
  }

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box) const {
    return observe_fields_event_.get_observation_type_and_key_for_registration(
        box);
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ElementId<3>& /*element_id*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | observe_fields_event_;
    p | interpolate_event_;
  }

 private:
  ObserveFieldsEvent observe_fields_event_;
  InterpolateEvent interpolate_event_;
};

#if defined(SPECTRE_USE_CHARM)
/// \cond
// NOLINTBEGIN
template <size_t VolumeDim, typename InterpolationTargetTag,
          typename... InterpolatorSourceVarTags, typename... Tensors,
          typename... NonTensorComputeTags>
PUP::able::PUP_ID FindCommonHorizon<
    VolumeDim, InterpolationTargetTag, tmpl::list<InterpolatorSourceVarTags...>,
    tmpl::list<Tensors...>, tmpl::list<NonTensorComputeTags...>>::my_PUP_ID = 0;
// NOLINTEND
/// \endcond
#endif  // SPECTRE_USE_CHARM
}  // namespace intrp::Events

namespace ah::Events {
/*!
 * \brief Event that combines the `ah::Events::FindApparentHorizon` event with
 * `dg::Events::ObserveFields` specifically for common horizon finding.
 *
 * \details This is just a thin wrapper around each event and doesn't do
 * anything extra.
 */
template <typename HorizonMetavars, typename Tensors,
          typename NonTensorComputeTagsList = tmpl::list<>>
class FindCommonHorizon;

template <typename HorizonMetavars, typename... Tensors,
          typename... NonTensorComputeTags>
class FindCommonHorizon<HorizonMetavars, tmpl::list<Tensors...>,
                        tmpl::list<NonTensorComputeTags...>> : public Event {
  using ObserveFieldsEvent =
      dg::Events::ObserveFields<3, tmpl::list<Tensors...>,
                                tmpl::list<NonTensorComputeTags...>>;
  using HorizonFindEvent = ah::Events::FindApparentHorizon<HorizonMetavars>;

 public:
  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FindCommonHorizon);  // NOLINT
  /// \endcond

  using options = tmpl::append<typename ObserveFieldsEvent::options,
                               typename HorizonFindEvent::options>;
  static constexpr Options::String help =
      "Starts a common horizon find and sends the evolved variables to be "
      "written to disk.";

  static std::string name() { return pretty_type::name<HorizonMetavars>(); }

  FindCommonHorizon() = default;

  FindCommonHorizon(
      const std::string& subfile_name,
      FloatingPointType coordinates_floating_point_type,
      const std::vector<FloatingPointType>& floating_point_types,
      const std::vector<std::string>& variables_to_observe,
      std::optional<std::vector<std::string>> active_block_or_block_groups = {},
      std::optional<Mesh<3>> interpolation_mesh = {},
      std::optional<Mesh<3>> projection_mesh = {},
      const Options::Context& context = {})
      : observe_fields_event_(subfile_name, coordinates_floating_point_type,
                              floating_point_types, variables_to_observe,
                              active_block_or_block_groups, interpolation_mesh,
                              projection_mesh, std::nullopt, context),
        horizon_find_event_(subfile_name) {}

  using compute_tags_for_observation_box = tmpl::remove_duplicates<tmpl::append<
      typename ObserveFieldsEvent::compute_tags_for_observation_box,
      typename HorizonFindEvent::compute_tags_for_observation_box>>;

  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::ObservationBox, ::Events::Tags::ObserverMesh<3>,
                 domain::Tags::Element<3>>;

  template <typename DataBoxType, typename ComputeTagsList,
            typename Metavariables, typename ParallelComponent>
  void operator()(const ObservationBox<DataBoxType, ComputeTagsList>& box,
                  const Mesh<3>& mesh, const Element<3>& element,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<3>& element_id,
                  const ParallelComponent* const component,
                  const ObservationValue& observation_value) const {
    observe_fields_event_(box, mesh, cache, element_id, component,
                          observation_value);

    horizon_find_event_(
        get<typename HorizonMetavars::time_tag>(box), mesh, element,
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(box),
        get<gh::Tags::Pi<DataVector, 3>>(box),
        get<gh::Tags::Phi<DataVector, 3>>(box),
        get<::Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                          Frame::Inertial>>(box),
        cache, element_id, component, observation_value);
  }

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box) const {
    return observe_fields_event_.get_observation_type_and_key_for_registration(
        box);
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ElementId<3>& /*element_id*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | observe_fields_event_;
    p | horizon_find_event_;
  }

 private:
  ObserveFieldsEvent observe_fields_event_;
  HorizonFindEvent horizon_find_event_;
};

#if defined(SPECTRE_USE_CHARM)
/// \cond
// NOLINTBEGIN
template <typename HorizonMetavars, typename... Tensors,
          typename... NonTensorComputeTags>
PUP::able::PUP_ID
    FindCommonHorizon<HorizonMetavars, tmpl::list<Tensors...>,
                      tmpl::list<NonTensorComputeTags...>>::my_PUP_ID = 0;
// NOLINTEND
/// \endcond
#endif  // SPECTRE_USE_CHARM
}  // namespace ah::Events
