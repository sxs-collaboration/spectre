// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <ostream>

#include "ApparentHorizons/YlmSpherepackHelper.hpp"
#include "ErrorHandling/Error.hpp"

namespace YlmSpherepack_detail {

std::vector<double>& MemoryPool::get(size_t n_pts) noexcept {
  for (auto& elem : memory_pool_) {
    if (not elem.currently_in_use) {
      elem.currently_in_use = true;
      auto& temp = elem.storage;
      if (temp.size() < n_pts) {
        temp.resize(n_pts);
      }
      return temp;
    }
  }
  ERROR("Attempt to allocate more than " << num_temps_ << " temps.");
}

// We don't simply forward this to the const double* version
// because std::vector<T>::data() might be nullptr for vectors
// in the pool that are unallocated, for random vectors passed into
// this function that have nothing to do with the pool, or for vectors
// that are in the pool but have been size-zero allocated.
void MemoryPool::free(const std::vector<double>& to_be_freed) noexcept {
  for (auto& elem : memory_pool_) {
    if (&(elem.storage) == &to_be_freed) {
      elem.currently_in_use = false;
      return;
    }
  }
  ERROR("Attempt to free temp that was never allocated.");
}

/// \cond DOXYGEN_FAILS_TO_PARSE_THIS
void MemoryPool::free(const gsl::not_null<double*> to_be_freed) noexcept {
  for (auto& elem : memory_pool_) {
    if (elem.storage.data() == to_be_freed) {
      elem.currently_in_use = false;
      return;
    }
  }
  ERROR("Attempt to free temp that was never allocated.");
}
/// \endcond

}  // namespace YlmSpherepack_detail
