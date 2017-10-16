// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define metafunctions that produce Tags from other Tags

#pragma once

#include "Utilities/TMPL.hpp"

/// \ingroup DataBoxTags
/// \brief Create a new list of Tags by wrapping each tag in `TagList` using the
/// `Wrapper`.
template <template <typename...> class Wrapper, typename TagList,
          typename... Args>
using wrap_tags_in =
    tmpl::transform<TagList, tmpl::bind<Wrapper, tmpl::_1, Args...>>;
