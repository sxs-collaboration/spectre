// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/ElementId.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeStepId;
namespace Tags {
struct TimeStepId;
}  // namespace Tags
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
template <size_t Dim> class Mesh;
template <size_t VolumeDim> class ElementIndex;
namespace intrp {
template <typename Metavariables>
struct Interpolator;
namespace Actions {
struct InterpolatorReceiveVolumeData;
}  // namespace Actions
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Events {
template <size_t VolumeDim, typename Tensors, typename EventRegistrars>
class Interpolate;

namespace Registrars {
template <size_t VolumeDim, typename Tensors>
struct Interpolate {
  template <typename RegistrarList>
  using f = Events::Interpolate<VolumeDim, Tensors, RegistrarList>;
};
}  // namespace Registrars

/// Does an interpolation by calling Actions on Interpolator.
template <size_t VolumeDim, typename Tensors,
          typename EventRegistrars =
              tmpl::list<Registrars::Interpolate<VolumeDim, Tensors>>>
class Interpolate;  // IWYU pragma: keep

template <size_t VolumeDim, typename... Tensors, typename EventRegistrars>
class Interpolate<VolumeDim, tmpl::list<Tensors...>, EventRegistrars>
    : public Event<EventRegistrars> {
  /// \cond
  explicit Interpolate(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Interpolate);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help =
      "Starts interpolation by sending data to the Interpolator.";

  Interpolate() = default;

  using argument_tags =
      tmpl::list<::Tags::TimeStepId, domain::Tags::Mesh<VolumeDim>, Tensors...>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(const TimeStepId& time_id, const Mesh<VolumeDim>& mesh,
                  const db::const_item_type<Tensors>&... tensors,
                  Parallel::ConstGlobalCache<Metavariables>& cache,
                  const ElementIndex<VolumeDim>& array_index,
                  const ParallelComponent* const /*meta*/) const noexcept {
    Variables<tmpl::list<Tensors...>> interp_vars(mesh.number_of_grid_points());
    const auto copy_to_variables = [&interp_vars](const auto tensor_tag_v,
                                                  const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      get<tensor_tag>(interp_vars) = tensor;
      return 0;
    };
    expand_pack(copy_to_variables(tmpl::type_<Tensors>{}, tensors)...);

    // Send volume data to the Interpolator, to trigger interpolation.
    auto& interpolator =
        *::Parallel::get_parallel_component<Interpolator<Metavariables>>(cache)
             .ckLocalBranch();
    Parallel::simple_action<Actions::InterpolatorReceiveVolumeData>(
        interpolator, time_id, ElementId<VolumeDim>(array_index), mesh,
        interp_vars);
  }
};

/// \cond
template <size_t VolumeDim, typename... Tensors, typename EventRegistrars>
PUP::able::PUP_ID Interpolate<VolumeDim, tmpl::list<Tensors...>,
                              EventRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Events
}  // namespace intrp
