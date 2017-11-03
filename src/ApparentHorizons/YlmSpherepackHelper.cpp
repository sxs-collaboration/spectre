// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/YlmSpherepackHelper.hpp"
#include "ErrorHandling/Error.hpp"

namespace YlmSpherepack_detail {

std::vector<double>& MemoryPool::get(size_t n_pts) noexcept {
  for (auto& elem : memory_pool_) {
    if (!elem.currently_in_use) {
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

void MemoryPool::free(const std::vector<double>& to_be_freed) noexcept {
  bool found = false;
  for (auto& elem : memory_pool_) {
    if (&(elem.storage) == &to_be_freed) {
      elem.currently_in_use = false;
      found = true;
      break;
    }
  }
  if (!found) {
    ERROR("Attempt to free temp that was never allocated.");
  }
}
void MemoryPool::free(const double* to_be_freed) noexcept {
  bool found = false;
  for (auto& elem : memory_pool_) {
    if (elem.storage.data() == to_be_freed) {
      elem.currently_in_use = false;
      found = true;
      break;
    }
  }
  if (!found) {
    ERROR("Attempt to free temp that was never allocated.");
  }
}
}  // namespace YlmSpherepack_detail
