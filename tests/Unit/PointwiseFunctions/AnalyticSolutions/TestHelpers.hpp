// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace TestHelpers {
/// \ingroup TestingFrameworkGroup
/// Functions for testing analytic solutions
namespace AnalyticSolutions {
/// Checks that tags can be retrieved both individually and all at
/// once.
template <typename Solution, typename Coords, typename TagsList>
void test_tag_retrieval(const Solution& solution, const Coords& coords,
                        const double time, const TagsList /*meta*/) noexcept {
  const auto vars_from_all_tags = solution.variables(coords, time, TagsList{});
  const auto vars_from_all_tags_reversed =
      solution.variables(coords, time, tmpl::reverse<TagsList>{});
  tmpl::for_each<TagsList>([&](const auto tag_v) noexcept {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const auto single_var = solution.variables(coords, time, tmpl::list<tag>{});
    CHECK(tuples::get<tag>(single_var) == tuples::get<tag>(vars_from_all_tags));
    CHECK(tuples::get<tag>(single_var) ==
          tuples::get<tag>(vars_from_all_tags_reversed));
  });
}
}  // namespace AnalyticSolutions
}  // namespace TestHelpers
