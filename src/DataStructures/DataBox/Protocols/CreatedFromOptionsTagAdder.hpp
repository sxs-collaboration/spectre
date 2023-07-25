// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/OptionCreatableTag.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace db::protocols {
/// \brief Class that adds mutable DataBox items initialized from options.
///
/// A class conforming to this protocol can be used to add tags to a DataBox
/// that will be initialized from input file options.  The conforming class must
/// provide the following:
///
/// - `simple_tags_from_options`: A type list of tags conforming to
///   db::protocols::OptionCreatableTag.
///
/// \note Multiple conforming classes are allowed to add the same tag.
struct CreatedFromOptionsTagAdder {
  template <typename ConformingType>
  struct test {
    using simple_tags_from_options =
        typename ConformingType::simple_tags_from_options;
    static_assert(
        tmpl::all<simple_tags_from_options,
                  tt::assert_conforms_to<tmpl::_1, OptionCreatableTag>>::value);
  };
};
}  // namespace db::protocols
