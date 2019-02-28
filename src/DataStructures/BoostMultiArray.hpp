// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/config.hpp>
#include <cstddef>

// We need to change the index type from std::ptrdiff_t to std::size_t so that
// it matches the STL and we don't get unsigned cast warnings everywhere.
namespace boost {
namespace detail {
namespace multi_array{
using size_type = std::size_t;
using index = std::size_t;
} // namespace multi_array
} // namespace detail
} // namespace boost

#include <boost/multi_array.hpp>

