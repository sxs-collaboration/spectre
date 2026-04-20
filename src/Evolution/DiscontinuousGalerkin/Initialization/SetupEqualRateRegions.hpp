// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <charm++.h>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/EqualRateRegionId.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/EqualRateRegions.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Section.hpp"
#include "Parallel/Tags/Section.hpp"
#include "Parallel/TypeTraits.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionalIdMap;
template <size_t VolumeDim>
class Element;
class TimeDelta;
namespace Tags {
struct LtsMode;
struct TimeStep;
}  // namespace Tags
namespace domain::Tags {
template <size_t VolumeDim>
struct Element;
template <size_t Dim>
struct InitialRefinementLevels;
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
template <size_t Dim>
struct SetEqualRateSection {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      Parallel::Section<ParallelComponent, Tags::EqualRateRegionId> section) {
    const auto& equal_rate_regions =
        db::get<evolution::dg::Tags::EqualRateRegions<Dim>>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    if (equal_rate_regions.is_in_region(section.id(), element.id())) {
      db::mutate<
          Parallel::Tags::Section<ParallelComponent, Tags::EqualRateRegionId>>(
          [&](const gsl::not_null<std::optional<Parallel::Section<
                  ParallelComponent, Tags::EqualRateRegionId>>*>
                  box_section) {
            ASSERT(not box_section->has_value(),
                   "Already have an equal-rate section");
            box_section->emplace(std::move(section));
          },
          make_not_null(&box));
    }
  }
};

namespace SetupEqualRateRegions_detail {
template <size_t Dim>
struct SectionsToCreate {
  using argument_tags = tmpl::list<domain::Tags::Element<Dim>,
                                   evolution::dg::Tags::EqualRateRegions<Dim>,
                                   domain::Tags::InitialRefinementLevels<Dim>>;

  static std::vector<std::pair<EqualRateRegionId, std::vector<CkArrayIndex>>>
  apply(const Element<Dim>& element,
        const EqualRateRegionsBase<Dim>& equal_rate_regions,
        const std::vector<std::array<size_t, Dim>>& initial_refinement_levels);
};
}  // namespace SetupEqualRateRegions_detail

/// Set up `FixedLtsRatio` and mortar time-stepping policies to
/// disable local time-stepping in equal rate regions.
template <typename Metavariables, size_t Dim, typename RegionGenerators>
struct SetupEqualRateRegions {
 private:
  using array_components = tmpl::filter<typename Metavariables::component_list,
                                        Parallel::is_array<tmpl::_1>>;
  static_assert(tmpl::size<array_components>::value == 1,
                "Cannot identify element array.");

 public:
  using ParallelComponent = tmpl::front<array_components>;

  using const_global_cache_tags = tmpl::list<
      evolution::dg::Tags::ConcreteEqualRateRegions<Dim, RegionGenerators>>;
  using simple_tags = tmpl::list<
      ::Tags::FixedLtsRatio,
      Parallel::Tags::Section<ParallelComponent, Tags::EqualRateRegionId>>;
  using compute_tags = tmpl::list<
      evolution::dg::Tags::EqualRateRegionsRef<Dim, RegionGenerators>>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::get<::Tags::LtsMode>(box) == LtsMode::Off) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    db::mutate_apply<SetupLocalEqualRateRegion<Dim>>(make_not_null(&box));

    // Set up sections.  This will be empty everywhere except element 0.
    const auto sections_to_create =
        db::apply<SetupEqualRateRegions_detail::SectionsToCreate<Dim>>(box);
    for (const auto& [region_id, section_elements] : sections_to_create) {
      auto& component =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      using Section =
          Parallel::Section<ParallelComponent, Tags::EqualRateRegionId>;
      const Section section{
          region_id, Section::cproxy_section::ckNew(component.ckGetArrayID(),
                                                    section_elements)};

      // Charm sections are buggy.  Most of their features only work
      // when manually sending charm messages.  The higher-level
      // interfaces exist, but fail to actually transmit data to the
      // receiving end and pass in garbage instead.  This is mentioned
      // in an offhand remark in a section of the charm documentation
      // on "optimized multicast".  As a result we avoid doing
      // anything other than reductions (which seem to work) on
      // sections.  Here we use an array broadcast and ignore the
      // message on most of the elements because section broadcasts
      // don't work.
      Parallel::simple_action<SetEqualRateSection<Dim>>(component, section);
    }

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace evolution::dg::Initialization
