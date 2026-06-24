// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/EqualRateRegions.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionalIdMap;
template <size_t VolumeDim>
class Element;
class TimeDelta;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct TimeStep;
}  // namespace Tags
namespace domain::Tags {
template <size_t VolumeDim>
struct Element;
}  // namespace domain::Tags
namespace evolution::dg {
template <size_t Dim>
class EqualRateRegionsBase;
template <size_t VolumeDim>
class MortarInfo;
}  // namespace evolution::dg
namespace evolution::dg::Tags {
template <size_t Dim>
struct MortarInfo;
}  // namespace evolution::dg::Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::Initialization {
/// Set up `FixedLtsRatio` and mortar time-stepping policies to
/// disable local time-stepping in equal rate regions.
template <size_t Dim>
struct SetupLocalEqualRateRegion {
  using return_tags =
      tmpl::list<::Tags::FixedLtsRatio, evolution::dg::Tags::MortarInfo<Dim>>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>,
                 evolution::dg::Tags::EqualRateRegions<Dim>, ::Tags::TimeStep>;

  static void apply(
      gsl::not_null<std::optional<size_t>*> fixed_lts_ratio,
      gsl::not_null<DirectionalIdMap<Dim, ::evolution::dg::MortarInfo<Dim>>*>
          mortar_infos,
      const Element<Dim>& element,
      const EqualRateRegionsBase<Dim>& equal_rate_regions,
      const TimeDelta& time_step);
};

namespace Actions {
/// Set up `FixedLtsRatio` and mortar time-stepping policies to
/// disable local time-stepping in equal rate regions.
template <size_t Dim, typename RegionGenerators>
struct SetupEqualRateRegions {
  using const_global_cache_tags = tmpl::list<
      evolution::dg::Tags::ConcreteEqualRateRegions<Dim, RegionGenerators>>;
  using simple_tags = tmpl::list<::Tags::FixedLtsRatio>;
  using compute_tags = tmpl::list<
      evolution::dg::Tags::EqualRateRegionsRef<Dim, RegionGenerators>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<SetupLocalEqualRateRegion<Dim>>(make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace evolution::dg::Initialization
