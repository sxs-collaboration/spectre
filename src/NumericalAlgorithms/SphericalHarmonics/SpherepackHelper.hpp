// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"

namespace ylm::Spherepack_detail {

/// Holds the various constant 'work' arrays for Spherepack.
/// These are computed only once during ylm::Spherepack construction
/// and are re-used over and over again.
class ConstStorage {
 public:
  ConstStorage(const size_t l_max, const size_t m_max);
  ~ConstStorage() = default;
  ConstStorage(const ConstStorage& rhs);
  ConstStorage& operator=(const ConstStorage& rhs);
  ConstStorage(ConstStorage&& rhs);
  ConstStorage& operator=(ConstStorage&& rhs);

  std::vector<size_t> work_interp_index;
  // The following are vectors because they are returnable by
  // member functions.
  std::vector<double> quadrature_weights, theta, phi;
  // All other storage is allocated in a single DataVector and then
  // pointed to by gsl::spans.
  DataVector storage_;
  gsl::span<double> work_phys_to_spec, work_scalar_spec_to_phys;
  gsl::span<double> work_vector_spec_to_phys;
  gsl::span<double> sin_theta, cos_theta, cosec_theta, cot_theta;
  gsl::span<double> sin_phi, cos_phi;
  gsl::span<double> work_interp_alpha, work_interp_beta, work_interp_pmm;

 private:
  size_t l_max_;
  size_t m_max_;
  void point_spans_to_data_vector(const bool allocate = false);
};

/// This is a quick way of providing temporary space that is
/// re-utilized many times without the expense of mallocs.  This
/// turned out to be important for optimizing SpEC (because
/// ylm::Spherepack is used for the basis functions in the most
/// expensive subdomains) but may or may not be important for SpECTRE.
class MemoryPool {
 public:
  MemoryPool() = default;
  std::vector<double>& get(size_t n_pts);
  void free(const std::vector<double>& to_be_freed);
  void free(gsl::not_null<double*> to_be_freed);
  void clear();

 private:
  struct StorageAndAvailability {
    std::vector<double> storage;
    bool currently_in_use{false};
  };
  // It turns out that the maximum number of temporaries needed in a
  // single ylm::Spherepack calculation is 9.
  static const constexpr size_t num_temps_ = 9;
  std::array<StorageAndAvailability, num_temps_> memory_pool_;
};
}  // namespace ylm::Spherepack_detail
