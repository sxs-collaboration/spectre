// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/TagName.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace db {

/*!
 * \brief Validate that the selected names are a subset of the given tags.
 *
 * The possible choices for the selection are the `db::tag_name`s of the `Tags`.
 *
 * \tparam Tags List of possible tags.
 * \param selected_names Names to validate.
 * \param context Options context for error reporting.
 */
template <typename Tags>
void validate_selection(const std::vector<std::string>& selected_names,
                        const Options::Context& context) {
  using ::operator<<;
  std::unordered_set<std::string> valid_names{};
  tmpl::for_each<Tags>([&valid_names](auto tag_v) {
    using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
    valid_names.insert(db::tag_name<tag>());
  });
  for (const auto& name : selected_names) {
    if (valid_names.find(name) == valid_names.end()) {
      PARSE_ERROR(context, "Invalid selection: " << name
                                                 << ". Possible choices are: "
                                                 << valid_names << ".");
    }
    if (alg::count(selected_names, name) != 1) {
      PARSE_ERROR(context, name << " specified multiple times");
    }
  }
}

}  // namespace db
