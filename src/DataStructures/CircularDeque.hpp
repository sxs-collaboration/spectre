// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/StaticDeque.hpp"

// This class is tested in Test_StaticDeque.cpp

/// A class implementing the std::deque interface using a circular
/// buffer to avoid allocations when the size does not exceed a
/// previous allocated capacity.
///
/// The class is optimized for a small number of elements with many
/// balanced insertions and removals.  As such, the capacity is not
/// increased beyond the size required when inserting elements in
/// order to save memory in the steady-state.
///
/// Differences from std::deque:
/// * Insertions (including during construction) are O(n) if the
///   previous capacity is exceeded and invalidate all references and
///   iterators.  Some cases where multiple insertions happen in the
///   same method are optimized to perform only one reallocation.
/// * Erasing elements from the front of the queue invalidates all
///   iterators (but not references).
///
/// This last point is not a fundamental limitation, but could be
/// corrected with a more complicated iterator implementation if the
/// standard behavior is found to be useful.
///
/// \note This class does not behave like a standard circular buffer,
/// in that insertion operations never overwrite existing elements.
/// The circularness is only a reference to the implementation using
/// circular internal storage to avoid allocations.
template <typename T>
class CircularDeque
    : public StaticDeque<T, std::numeric_limits<size_t>::max()> {
  using StaticDeque<T, std::numeric_limits<size_t>::max()>::StaticDeque;
};
