// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/SetupEqualRateRegions.hpp"

#include <array>
#include <charm++.h>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/FaceType.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Time/Time.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::Initialization {
template <size_t Dim>
void SetupLocalEqualRateRegion<Dim>::apply(
    const gsl::not_null<std::optional<size_t>*> fixed_lts_ratio,
    const gsl::not_null<
        DirectionalIdMap<Dim, ::evolution::dg::MortarInfo<Dim>>*>
        mortar_infos,
    const Element<Dim>& element,
    const EqualRateRegionsBase<Dim>& equal_rate_regions,
    const TimeDelta& time_step) {
  std::optional<EqualRateRegionId> my_region{};
  for (const auto& region : equal_rate_regions.regions()) {
    if (equal_rate_regions.is_in_region(region.second, element.id())) {
      if (my_region.has_value()) {
        ERROR(
            "Element " << element.id() << " is in multiple equal-rate regions: "
                       << equal_rate_regions.region_names().at(*my_region)
                       << " and "
                       << equal_rate_regions.region_names().at(region.second));
      }
      my_region.emplace(region.second);
    }
  }

  if (not my_region.has_value()) {
    return;
  }

  fixed_lts_ratio->emplace(time_step.fraction().denominator());
  for (auto& entry : *mortar_infos) {
    const auto& mortar_id = entry.first;
    auto& info = entry.second;
    bool neighbor_is_in_region{};
    if (element.face_types().at(mortar_id.direction()) !=
        domain::FaceType::MultipleNonconforming) {
      neighbor_is_in_region =
          equal_rate_regions.is_in_region(*my_region, mortar_id.id());
    } else {
      const auto& neighbors =
          element.neighbors().at(mortar_id.direction()).ids();
      if (neighbors.empty()) {
        ERROR("Have nonconforming mortar with no neighbors");
      }
      auto neighbor = neighbors.begin();
      neighbor_is_in_region =
          equal_rate_regions.is_in_region(*my_region, *neighbor);
      ++neighbor;
      for (; neighbor != neighbors.end(); ++neighbor) {
        if (equal_rate_regions.is_in_region(*my_region, *neighbor) !=
            neighbor_is_in_region) {
          ERROR(
              "Element " << element.id() << " is in equal-rate region "
                         << equal_rate_regions.region_names().at(*my_region)
                         << ", but only some neighbors across mortar "
                         << mortar_id
                         << " are.  Cannot find a valid time-stepping policy.");
        }
      }
    }

    if (neighbor_is_in_region) {
      info.time_stepping_policy() = TimeSteppingPolicy::EqualRate;
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template struct SetupLocalEqualRateRegion<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

namespace Actions::SetupEqualRateRegions_detail {
template <size_t Dim>
std::vector<std::pair<EqualRateRegionId, std::vector<CkArrayIndex>>>
SectionsToCreate<Dim>::apply(
    const Element<Dim>& element,
    const EqualRateRegionsBase<Dim>& equal_rate_regions,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels) {
  if (not is_zeroth_element(element.id())) {
    // Only one element should create the sections.
    return {};
  }

  const auto& regions = equal_rate_regions.regions();

  std::vector<std::pair<EqualRateRegionId, std::vector<CkArrayIndex>>>
      sections{};
  sections.reserve(regions.size());

  const auto all_elements = initial_element_ids(initial_refinement_levels);
  for (const auto& [name, region_id] : regions) {
    std::vector<CkArrayIndex> section_elements{};
    for (const ElementId<Dim>& element_id : all_elements) {
      if (equal_rate_regions.is_in_region(region_id, element_id)) {
        section_elements.push_back(Parallel::ArrayIndex(element_id));
      }
    }
    sections.emplace_back(region_id, std::move(section_elements));
  }
  return sections;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template struct SectionsToCreate<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace Actions::SetupEqualRateRegions_detail
}  // namespace evolution::dg::Initialization
