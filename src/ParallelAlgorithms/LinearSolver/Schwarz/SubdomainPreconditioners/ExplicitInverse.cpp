// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainPreconditioners/ExplicitInverse.hpp"

#include <cstddef>
#include <stdexcept>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace LinearSolver::Schwarz::subdomain_preconditioners {

template <size_t Dim>
void ExplicitInverse<Dim>::invert_matrix() noexcept {
  try {
    invert(inverse_);
  } catch (const std::invalid_argument& e) {
    ERROR("Could not invert subdomain matrix (size " << size_
                                                     << "): " << e.what());
  }
}

template <size_t Dim>
void ExplicitInverse<Dim>::apply_inverse() const noexcept {
  result_workspace_ = inverse_ * arg_workspace_;
}

template <size_t Dim>
size_t ExplicitInverse<Dim>::size() const noexcept {
  return size_;
}

template <size_t Dim>
DenseMatrix<double> ExplicitInverse<Dim>::matrix_representation() const
    noexcept {
  return inverse_;
}

template <size_t Dim>
void ExplicitInverse<Dim>::pup(PUP::er& p) noexcept {
  p | ordered_overlap_ids_;
  p | inverse_;
  p | size_;
  if (p.isUnpacking() and size_ != std::numeric_limits<size_t>::max()) {
    arg_workspace_.resize(size_);
    result_workspace_.resize(size_);
  }
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data) template struct ExplicitInverse<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace LinearSolver::Schwarz::subdomain_preconditioners
