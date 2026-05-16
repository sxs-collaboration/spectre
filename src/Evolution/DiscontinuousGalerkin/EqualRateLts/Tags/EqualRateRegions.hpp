// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Tags {
/// Regions that cannot perform local time-stepping.
///
/// Accessible as a base class through the `EqualRateRegions` tag
/// without the second template parameter.
template <size_t Dim, typename RegionGenerators>
struct ConcreteEqualRateRegions : db::SimpleTag {
  using type = evolution::dg::EqualRateRegions<Dim, RegionGenerators>;
  using option_tags = typename type::creation_tags;

  static constexpr bool pass_metavariables = false;
  template <typename... Args>
  static type create_from_options(const Args&... options) {
    return type{options...};
  }
};

/// Regions that cannot perform local time-stepping.
template <size_t Dim>
struct EqualRateRegions : db::SimpleTag {
  using type = evolution::dg::EqualRateRegionsBase<Dim>;
};

/// Reference tag converting ConcreteEqualRateRegions to the base class.
template <size_t Dim, typename RegionGenerators>
struct EqualRateRegionsRef : EqualRateRegions<Dim>, db::ReferenceTag {
  using base = EqualRateRegions<Dim>;
  using argument_tags =
      tmpl::list<ConcreteEqualRateRegions<Dim, RegionGenerators>>;
  static const EqualRateRegionsBase<Dim>& get(
      const EqualRateRegionsBase<Dim>& regions) {
    return regions;
  }
};
}  // namespace evolution::dg::Tags
