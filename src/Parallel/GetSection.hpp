// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags/Section.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace Parallel {

/*!
 * \brief Retrieve the section that the element belongs to, or
 * `Parallel::no_section()` if `SectionIdTag` is `void`.
 *
 * This function is useful to support sections in parallel algorithms. Specify
 * the `SectionIdTag` template parameter to retrieve the associated section, or
 * set it to `void` when the parallel algorithm runs over all elements of the
 * parallel component. See `Parallel::Section` for details on sections.
 *
 * Only call this function on elements that are part of a section. In case not
 * all elements are part of a section with the `SectionIdTag`, make sure to skip
 * those elements before calling this function.
 */
template <typename ParallelComponent, typename SectionIdTag,
          typename DbTagsList>
auto& get_section([[maybe_unused]] const gsl::not_null<db::DataBox<DbTagsList>*>
                      box) noexcept {
  if constexpr (std::is_same_v<SectionIdTag, void>) {
    return Parallel::no_section();
  } else {
    std::optional<Parallel::Section<ParallelComponent, SectionIdTag>>& section =
        db::get_mutable_reference<
            Parallel::Tags::Section<ParallelComponent, SectionIdTag>>(box);
    ASSERT(section.has_value(),
           "Call 'get_section' only on elements that belong to a section. This "
           "is probably a bug, because the other elements should presumably be "
           "skipped. The section is: "
               << db::tag_name<SectionIdTag>());
    return *section;
  }
}

}  // namespace Parallel
