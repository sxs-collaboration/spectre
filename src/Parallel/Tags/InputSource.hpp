// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Tags.hpp"

namespace Parallel::Tags {
/// \ingroup DataBoxTagsGroup
/// A tag storing the input file yaml input source passed in at runtime, so that
/// it can be accessed and written to files.
struct InputSource : db::SimpleTag {
  using type = std::vector<std::string>;
  using option_tags = tmpl::list<::Options::Tags::InputSource>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& input_source) {
    return input_source;
  }
};
}  // namespace Parallel::Tags
