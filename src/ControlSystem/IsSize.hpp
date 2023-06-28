// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ControlSystem/Systems/Size.hpp"
#include "Domain/Structure/ObjectLabel.hpp"

namespace control_system::size {
// tt::is_a doesn't work because of domain::ObjectLabel and size_t
template <typename T>
struct is_size : std::false_type {};

template <domain::ObjectLabel Label, size_t DerivOrder>
struct is_size<Systems::Size<Label, DerivOrder>> : std::true_type {};

/// Check whether a control system is the `control_system::Systems::Size` system
/// or not
template <typename T>
constexpr bool is_size_v = is_size<T>::value;
}  // namespace control_system::size
