// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <concepts>
#include <pup.h>

/// Concept for serialization via pup.
template <typename T>
concept serializable =
    std::default_initializable<T> and requires(T t, PUP::er& p) { p | t; };
