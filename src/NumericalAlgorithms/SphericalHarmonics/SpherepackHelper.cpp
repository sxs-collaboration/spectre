// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <ostream>

#include "NumericalAlgorithms/SphericalHarmonics/SpherepackHelper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace ylm::Spherepack_detail {

ConstStorage::ConstStorage(const size_t l_max, const size_t m_max)
    : l_max_(l_max), m_max_(m_max) {
  point_spans_to_data_vector(true);
}

ConstStorage::ConstStorage(const ConstStorage& rhs)
    : work_interp_index(rhs.work_interp_index),
      quadrature_weights(rhs.quadrature_weights),
      theta(rhs.theta),
      phi(rhs.phi),
      storage_(rhs.storage_),
      l_max_(rhs.l_max_),
      m_max_(rhs.m_max_) {
  point_spans_to_data_vector();
}

ConstStorage& ConstStorage::operator=(const ConstStorage& rhs) {
  l_max_ = rhs.l_max_;
  m_max_ = rhs.m_max_;
  work_interp_index = rhs.work_interp_index;
  quadrature_weights = rhs.quadrature_weights;
  theta = rhs.theta;
  phi = rhs.phi;
  storage_ = rhs.storage_;
  point_spans_to_data_vector();
  return *this;
}

ConstStorage::ConstStorage(ConstStorage&& rhs)
    : work_interp_index(std::move(rhs.work_interp_index)),
      quadrature_weights(std::move(rhs.quadrature_weights)),
      theta(std::move(rhs.theta)),
      phi(std::move(rhs.phi)),
      storage_(std::move(rhs.storage_)),
      l_max_(rhs.l_max_),
      m_max_(rhs.m_max_) {
  point_spans_to_data_vector();
}

ConstStorage& ConstStorage::operator=(ConstStorage&& rhs) {
  l_max_ = rhs.l_max_;
  m_max_ = rhs.m_max_;
  work_interp_index = std::move(rhs.work_interp_index);
  quadrature_weights = std::move(rhs.quadrature_weights);
  theta = std::move(rhs.theta);
  phi = std::move(rhs.phi);
  storage_ = std::move(rhs.storage_);
  point_spans_to_data_vector();
  return *this;
}

void ConstStorage::point_spans_to_data_vector(const bool allocate) {
  // Below note that n_theta, n_phi, l1, l2, and int_work_size are
  // ints, not size_ts. The reason for this: The last term in the
  // expression for int_work_size can sometimes be negative. If
  // evaluated using unsigned ints instead of ints, it can underflow
  // and give a huge number.
  const auto n_theta = static_cast<int>(l_max_ + 1);
  const auto n_phi = static_cast<int>(2 * m_max_ + 1);
  const auto l1 = static_cast<int>(m_max_ + 1);
  const auto l2 = (n_theta + 1) / 2;
  const int int_work_size = n_phi + 15 + n_theta * (3 * (l1 + l2) - 2) +
                            (l1 - 1) * (l2 * (2 * n_theta - l1) - 3 * l1) / 2;
  ASSERT(int_work_size >= 0, "Bad size " << int_work_size);
  const auto scalar_work_size = static_cast<size_t>(int_work_size);

  // For vectors, the things that SPHEREPACK internally calls l1, mdb,
  // and mdc have a minimum allowed size of
  // min(n_theta_,(n_phi_+1)/2)).  However, here we set those
  // quantities to min(n_theta_, (n_phi_+2)/2)) which is what
  // SPHEREPACK requires for what it internally calls l1 and mdab for
  // scalars.  This results in a simplification, but also a (very
  // slightly) larger vector work array than technically required, for
  // some choices of parameters.

  // NOLINTNEXTLINE bugprone-misplaced-widening-cast
  const auto vector_work_size = static_cast<size_t>(
      n_theta * l2 * (n_theta + 1) + n_phi + 15 + 2 * n_theta);
  // NOLINTNEXTLINE bugprone-misplaced-widening-cast
  const auto quad_work_size = static_cast<size_t>(n_theta * n_phi);
  const size_t theta_work_size = l_max_ + 1;
  const size_t phi_work_size = 2 * m_max_ + 1;
  const auto interp_work_size =
      static_cast<size_t>(n_theta * l1 - l1 * (l1 - 1) / 2);
  const size_t interp_work_pmm_size = m_max_ + 1;

  const size_t full_size = 2 * scalar_work_size + vector_work_size +
                           4 * theta_work_size + 2 * phi_work_size +
                           2 * interp_work_size + interp_work_pmm_size;

  if (allocate) {
    work_interp_index.resize(interp_work_size);
    quadrature_weights.resize(quad_work_size);
    theta.resize(theta_work_size);
    phi.resize(phi_work_size);
    storage_ = DataVector(full_size);
  }

  size_t offset = 0;
  auto set_up_span = [&offset, this](const size_t size) -> gsl::span<double> {
    auto result = gsl::make_span(this->storage_.data() + offset, size);
    offset += size;
    return result;
  };
  work_phys_to_spec = set_up_span(scalar_work_size);
  work_scalar_spec_to_phys = set_up_span(scalar_work_size);
  work_vector_spec_to_phys = set_up_span(vector_work_size);
  sin_theta = set_up_span(theta_work_size);
  cos_theta = set_up_span(theta_work_size);
  cosec_theta = set_up_span(theta_work_size);
  cot_theta = set_up_span(theta_work_size);
  sin_phi = set_up_span(phi_work_size);
  cos_phi = set_up_span(phi_work_size);
  work_interp_alpha = set_up_span(interp_work_size);
  work_interp_beta = set_up_span(interp_work_size);
  work_interp_pmm = set_up_span(interp_work_pmm_size);

  ASSERT(offset == full_size, "Bug in dividing up memory");
}

std::vector<double>& MemoryPool::get(size_t n_pts) {
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
void MemoryPool::free(const std::vector<double>& to_be_freed) {
  for (auto& elem : memory_pool_) {
    if (&(elem.storage) == &to_be_freed) {
      elem.currently_in_use = false;
      return;
    }
  }
  ERROR("Attempt to free temp that was never allocated.");
}

void MemoryPool::free(const gsl::not_null<double*> to_be_freed) {
  for (auto& elem : memory_pool_) {
    if (elem.storage.data() == to_be_freed) {
      elem.currently_in_use = false;
      return;
    }
  }
  ERROR("Attempt to free temp that was never allocated.");
}

void MemoryPool::clear() {
  for (auto& elem : memory_pool_) {
    if (elem.currently_in_use) {
      ERROR("Attempt to clear element that is in use");
    }
    elem.storage.clear();
  }
}

}  // namespace ylm::Spherepack_detail
