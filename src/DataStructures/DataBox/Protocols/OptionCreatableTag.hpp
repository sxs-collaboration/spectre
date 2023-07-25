// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/TagTraits.hpp"
#include "Utilities/TypeTraits/FunctionInfo.hpp"

namespace db::protocols {
/// \brief DataBox tag for a mutable item that can be created from Options
///
/// A class conforming to this protocol can be used as a DataBox tag for any
/// mutable item that can be created from an input file options.  In particular,
/// it can be used as a tag in the `simple_tags_from_options` tag list for any
/// class conforming to db::protocols::CreatedFromOptionsTagAdder`.  The
/// conforming class must provide the following:
///
/// - `option_tags`: A type list of OptionTags corresponding to objects that are
///   created from an Options::Parser
/// - `create_from_options`: A static function whose return type is that of
///   `ConformingType::type`, and that takes as arguments the types
///   corresponding to `OptionTag::type` for each `OptionTag` in `option_tags`
/// - `pass_metavariables`: A `static constexpr bool` that indicates whether or
///   not `option_tags` and `create_from_options` is templated on a
///   metavariables class
struct OptionCreatableTag {
  template <typename ConformingType>
  struct test {
    using option_tags = typename ConformingType::option_tags;
    static constexpr bool pass_metavariables =
        ConformingType::pass_metavariables;
    static_assert(db::is_simple_tag_v<ConformingType>);
    static_assert(
        std::is_same_v<
            typename ConformingType::type,
            typename tt::function_info<
                decltype(&ConformingType::create_from_options)>::return_type>);
  };
};
}  // namespace db::protocols
