// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// This code is in a separate file from `MathWrapper.hpp` to avoid including
// `Framework/TestingFramework.hpp` in the source file, which breaks some builds
// on macOS with Catch2 "undefined symbols" errors.

#include <type_traits>
#include <utility>

#include "DataStructures/MathWrapper.hpp"

namespace TestHelpers::MathWrapper {
// Defined and instantiated in a separate compilation unit to test the
// explicit instantiation code.
namespace detail {
template <typename T>
void do_assignment(const ::MathWrapper<T>& dest,
                   const ::MathWrapper<const T>& source);

template <typename T>
void do_multiply(const ::MathWrapper<T>& dest,
                 const typename ::MathWrapper<T>::scalar_type& scalar,
                 const ::MathWrapper<const T>& source);
}  // namespace detail
}  // namespace TestHelpers::MathWrapper
