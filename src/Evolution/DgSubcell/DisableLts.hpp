// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionalIdMap;
template <size_t VolumeDim>
class Element;
namespace domain::Tags {
template <size_t VolumeDim>
struct Element;
}  // namespace domain::Tags
namespace evolution::dg {
template <size_t VolumeDim>
class MortarInfo;
}  // namespace evolution::dg
namespace evolution::dg::Tags {
template <size_t Dim>
struct MortarInfo;
}  // namespace evolution::dg::Tags
namespace evolution::dg::subcell {
class SubcellOptions;
}  // namespace evolution::dg::subcell
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief Initialization mutator that disables local time-stepping in
 * subcell regions for a mixed subcell-LTS evolution.
 *
 * Sets `Tags::FixedLtsRatio` for all elements not specified in
 * `OnlyDgBlocksAndGroups`, and sets all boundaries between such
 * elements to `TimeSteppingPolicy::EqualRate`.
 */
template <size_t Dim>
struct DisableLts {
  using const_global_cache_tags = tmpl::list<Tags::SubcellOptions<Dim>>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags = tmpl::list<::Tags::FixedLtsRatio>;
  using compute_tags = tmpl::list<>;

  using return_tags =
      tmpl::list<::Tags::FixedLtsRatio, evolution::dg::Tags::MortarInfo<Dim>>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>, Tags::SubcellOptions<Dim>>;

  static void apply(
      gsl::not_null<std::optional<size_t>*> fixed_lts_ratio,
      gsl::not_null<DirectionalIdMap<Dim, ::evolution::dg::MortarInfo<Dim>>*>
          mortar_infos,
      const Element<Dim>& element, const SubcellOptions& subcell_options);
};
}  // namespace evolution::dg::subcell
