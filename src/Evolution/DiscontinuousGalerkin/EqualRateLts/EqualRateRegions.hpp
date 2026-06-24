// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>

#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::dg {
/// Unique identifier for an equal-rate region.
///
/// \see EqualRateRegions
struct EqualRateRegionId {
  size_t type;
  size_t label;

  EqualRateRegionId() = default;
  EqualRateRegionId(const size_t type_in, const size_t label_in)
      : type(type_in), label(label_in) {}

  void pup(PUP::er& p);

  friend auto operator<=>(const EqualRateRegionId&,
                          const EqualRateRegionId&) = default;
};

std::ostream& operator<<(std::ostream& os, const EqualRateRegionId& id);

namespace EqualRateRegions_detail {
template <typename T>
struct creation_tags {
  using type = typename T::creation_tags;
};
}  // namespace EqualRateRegions_detail

/// Regions of the domain that cannot perform local time-stepping.
///
/// The \p RegionGenerators template argument must be a `tmpl::list`
/// of classes satisfying the `equal_rate_region_generator<Dim>`
/// concept.
template <
    size_t Dim, typename RegionGenerators,
    typename CreationTags = tmpl::join<tmpl::transform<
        RegionGenerators, EqualRateRegions_detail::creation_tags<tmpl::_1>>>>
class EqualRateRegions;

/// \copydoc EqualRateRegions
template <size_t Dim>
class EqualRateRegionsBase {
 protected:
  EqualRateRegionsBase() = default;
  EqualRateRegionsBase(const EqualRateRegionsBase&) = default;
  EqualRateRegionsBase(EqualRateRegionsBase&&) = default;
  EqualRateRegionsBase& operator=(const EqualRateRegionsBase&) = default;
  EqualRateRegionsBase& operator=(EqualRateRegionsBase&&) = default;
  ~EqualRateRegionsBase() = default;

 public:
  /// Map from all region names to ids.  Inverse of `region_names()`.
  virtual const std::unordered_map<std::string, EqualRateRegionId>& regions()
      const = 0;

  /// Map from all region ids to names.  Inverse of `regions()`.
  virtual const std::map<EqualRateRegionId, std::string>& region_names()
      const = 0;

  /// Check whether a particular element is in a given region.
  virtual bool is_in_region(const EqualRateRegionId& region,
                            const ElementId<Dim>& element) const = 0;
};

/// \copydoc EqualRateRegions
template <size_t Dim, typename... RegionGenerators, typename... CreationTags>
class EqualRateRegions<Dim, tmpl::list<RegionGenerators...>,
                       tmpl::list<CreationTags...>>
    final : public EqualRateRegionsBase<Dim> {
 public:
  explicit EqualRateRegions()
    requires(sizeof...(CreationTags) > 0)
  = default;

  using creation_tags = tmpl::list<CreationTags...>;

  explicit EqualRateRegions(const typename CreationTags::type&... args);

  const std::unordered_map<std::string, EqualRateRegionId>& regions()
      const override;

  const std::map<EqualRateRegionId, std::string>& region_names() const override;

  bool is_in_region(const EqualRateRegionId& region,
                    const ElementId<Dim>& element) const override;

  void pup(PUP::er& p);

 private:
  std::tuple<RegionGenerators...> generators_{};
  std::unordered_map<std::string, EqualRateRegionId> regions_{};
  std::map<EqualRateRegionId, std::string> region_names_{};
};
}  // namespace evolution::dg
