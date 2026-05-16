// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"

#include <cstddef>
#include <map>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegionGenerator.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/SplitTuple.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
template <size_t Dim, typename... RegionGenerators, typename... CreationTags>
EqualRateRegions<Dim, tmpl::list<RegionGenerators...>,
                 tmpl::list<CreationTags...>>::
    EqualRateRegions(const typename CreationTags::type&... args) {
  // Put this in the first function.  Hopefully the compiler
  // instantiates things in order and it gets checked early.
  static_assert((... and equal_rate_region_generator<RegionGenerators, Dim>));

  const auto generator_args = split_tuple<
      tmpl::size<typename RegionGenerators::creation_tags>::value...>(
      std::forward_as_tuple(args...));
  tmpl::for_each<tmpl::range<size_t, 0, sizeof...(RegionGenerators)>>(
      [&]<size_t N>(tmpl::type_<tmpl::size_t<N>> /*meta*/) {
        get<N>(generators_) = std::apply(
            [](const auto&... gen_args) {
              return std::tuple_element_t<N, decltype(generators_)>(
                  gen_args...);
            },
            get<N>(generator_args));

        auto new_regions = get<N>(generators_).regions();
        while (not new_regions.empty()) {
          auto region = new_regions.extract(new_regions.begin());
          const bool inserted =
              regions_.try_emplace(std::move(region.key()), N, region.mapped())
                  .second;
          if (not inserted) {
            ERROR("Generated multiple regions named " << region.key());
          }
        }
      });

  for (const auto& [name, region] : regions_) {
    region_names_.try_emplace(region, name);
  }
}

template <size_t Dim, typename... RegionGenerators, typename... CreationTags>
const std::unordered_map<std::string, EqualRateRegionId>&
EqualRateRegions<Dim, tmpl::list<RegionGenerators...>,
                 tmpl::list<CreationTags...>>::regions() const {
  return regions_;
}

template <size_t Dim, typename... RegionGenerators, typename... CreationTags>
const std::map<EqualRateRegionId, std::string>&
EqualRateRegions<Dim, tmpl::list<RegionGenerators...>,
                 tmpl::list<CreationTags...>>::region_names() const {
  return region_names_;
}

template <size_t Dim, typename... RegionGenerators, typename... CreationTags>
bool EqualRateRegions<Dim, tmpl::list<RegionGenerators...>,
                      tmpl::list<CreationTags...>>::
    is_in_region(const EqualRateRegionId& region,
                 const ElementId<Dim>& element) const {
  return [&]<size_t... N>(std::index_sequence<N...> /*meta*/) {
    return (... or (region.type == N and
                    get<N>(generators_).is_in_region(region.label, element)));
  }(std::make_index_sequence<sizeof...(RegionGenerators)>{});
}

template <size_t Dim, typename... RegionGenerators, typename... CreationTags>
void EqualRateRegions<Dim, tmpl::list<RegionGenerators...>,
                      tmpl::list<CreationTags...>>::pup(PUP::er& p) {
  p | generators_;
  p | regions_;
  p | region_names_;
}
}  // namespace evolution::dg
