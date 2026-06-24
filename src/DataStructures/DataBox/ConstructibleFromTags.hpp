// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace db {
/// Concept for a class constructible from tags listed in a
/// `creation_tags` type alias.
template <typename T>
concept constructible_from_tags = tmpl::wrap<
    tmpl::push_front<tmpl::transform<typename T::creation_tags,
                                     tmpl::bind<tmpl::type_from, tmpl::_1>>,
                     T>,
    std::is_constructible>::value;
}  // namespace db
