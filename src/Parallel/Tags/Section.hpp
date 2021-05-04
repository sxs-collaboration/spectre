// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Parallel/Section.hpp"

namespace Parallel::Tags {

/*!
 * \brief The `Parallel::Section<ParallelComponent, SectionIdTag>` that this
 * element belongs to.
 *
 * The tag holds a `std::nullopt` if the element doesn't belong to a section of
 * the `SectionIdTag`. For example, a section may describe a specific region in
 * the computational domain and thus doesn't include all elements, so this tag
 * would hold `std::nullopt` on the elements outside that region.
 *
 * \see Parallel::Section
 */
template <typename ParallelComponent, typename SectionIdTag>
struct Section : db::SimpleTag {
  static std::string name() noexcept {
    return "Section(" + db::tag_name<SectionIdTag>() + ")";
  }
  using type =
      std::optional<Parallel::Section<ParallelComponent, SectionIdTag>>;
  // Allow default-constructing the tag so sections can be created and assigned
  // to the tag in a parallel component's `allocate_array` function.
  constexpr static bool pass_metavariables = false;
  using option_tags = tmpl::list<>;
  static type create_from_options() noexcept { return {}; };
};

}  // namespace Parallel::Tags
