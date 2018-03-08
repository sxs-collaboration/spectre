// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Utilities/Gsl.hpp"

namespace YlmSpherepack_detail {

/// Holds the various 'work' arrays for YlmSpherepack.
struct Storage {
  std::vector<double> work_phys_to_spec;
  std::vector<double> work_scalar_spec_to_phys;
  std::vector<double> work_vector_spec_to_phys;
  std::vector<double> theta, phi, sin_theta, cos_theta;
  std::vector<double> cos_phi, sin_phi, cosec_theta, cot_theta;
  std::vector<double> quadrature_weights;
  std::vector<double> work_interp_alpha;
  std::vector<double> work_interp_beta;
  std::vector<double> work_interp_pmm;
  std::vector<size_t> work_interp_index;
};

/// This is a quick way of providing temporary space that is
/// re-utilized many times without the expense of mallocs.  This
/// turned out to be important for optimizing SpEC (because
/// YlmSpherepack is used for the basis functions in the most
/// expensive subdomains) but may or may not be important for SpECTRE.
class MemoryPool {
 public:
  MemoryPool() = default;
  std::vector<double>& get(size_t n_pts) noexcept;
  void free(const std::vector<double>& to_be_freed) noexcept;
  void free(gsl::not_null<double*> to_be_freed) noexcept;

 private:
  struct StorageAndAvailability {
    std::vector<double> storage;
    bool currently_in_use{false};
  };
  // It turns out that the maximum number of temporaries needed in a
  // single YlmSpherepack calculation is 9.
  static const constexpr size_t num_temps_ = 9;
  std::array<StorageAndAvailability, num_temps_> memory_pool_;
};
}  // namespace YlmSpherepack_detail
