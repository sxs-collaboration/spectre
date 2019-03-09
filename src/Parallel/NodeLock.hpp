// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <converse.h>

#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief Create a converse CmiNodeLock
 */
inline CmiNodeLock create_lock() noexcept { return CmiCreateLock(); }

/*!
 * \ingroup ParallelGroup
 * \brief Free a converse CmiNodeLock. Using the lock after free is undefined
 * behavior.
 */
inline void free_lock(const gsl::not_null<CmiNodeLock*> node_lock) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiDestroyLock(*node_lock);
#pragma GCC diagnostic pop
}

/*!
 * \ingroup ParallelGroup
 * \brief Lock a converse CmiNodeLock
 */
inline void lock(const gsl::not_null<CmiNodeLock*> node_lock) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiLock(*node_lock);
#pragma GCC diagnostic pop
}

/// \cond
constexpr inline void lock(
    const gsl::not_null<NoSuchType*> /*unused*/) noexcept {}
/// \endcond

/*!
 * \ingroup ParallelGroup
 * \brief Returns true if the lock was successfully acquired and false if the
 * lock is already acquired by another processor.
 */
inline bool try_lock(const gsl::not_null<CmiNodeLock*> node_lock) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return CmiTryLock(*node_lock) == 0;
#pragma GCC diagnostic pop
}

/*!
 * \ingroup ParallelGroup
 * \brief Unlock a converse CmiNodeLock
 */
inline void unlock(const gsl::not_null<CmiNodeLock*> node_lock) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiUnlock(*node_lock);
#pragma GCC diagnostic pop
}

/// \cond
constexpr inline void unlock(
    const gsl::not_null<NoSuchType*> /*unused*/) noexcept {}
/// \endcond
}  // namespace Parallel
