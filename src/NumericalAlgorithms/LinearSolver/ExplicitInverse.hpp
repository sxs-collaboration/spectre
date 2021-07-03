// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/LinearSolver.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace LinearSolver::Serial {

/// \cond
template <typename LinearSolverRegistrars>
struct ExplicitInverse;
/// \endcond

namespace Registrars {
/// Registers the `LinearSolver::Serial::ExplicitInverse` linear solver
using ExplicitInverse = Registration::Registrar<Serial::ExplicitInverse>;
}  // namespace Registrars

namespace detail {
CREATE_IS_CALLABLE(reset)
CREATE_IS_CALLABLE_V(reset)
}  // namespace detail

/*!
 * \brief Linear solver that builds a matrix representation of the linear
 * operator and inverts it directly
 *
 * This solver first constructs an explicit matrix representation by "sniffing
 * out" the operator, i.e. feeding it with unit vectors, and then directly
 * inverts the matrix. The result is an operator that solves the linear problem
 * in a single step. This means that each element has a large initialization
 * cost, but all successive solves converge immediately.
 *
 * \par Advice on using this linear solver:
 *
 * - This solver is entirely agnostic to the structure of the linear operator.
 *   It is usually better to implement a linear solver that is specialized for
 *   your linear operator to take advantage of its properties. For example, if
 *   the operator has a tensor-product structure, the linear solver might take
 *   advantage of that. Only use this solver if no alternatives are available
 *   and if you have verified that it speeds up your solves.
 * - Since this linear solver stores the full inverse operator matrix it can
 *   have significant memory demands. For example, an operator representing a 3D
 *   first-order Elasticity system (9 variables) discretized on 12 grid points
 *   per dimension requires ca. 2GB of memory (per element) to store the matrix,
 *   scaling quadratically with the number of variables and with a power of 6
 *   with the number of grid points per dimension. Therefore, make sure to
 *   distribute the elements on a sufficient number of nodes to meet the memory
 *   requirements.
 * - This linear solver can be `reset()` when the operator changes (e.g. in each
 *   nonlinear-solver iteration). However, when using this solver as
 *   preconditioner it can be advantageous to avoid the reset and the
 *   corresponding cost of re-building the matrix and its inverse if the
 *   operator only changes "a little". In that case the preconditioner solves
 *   subdomain problems only approximately, but possibly still sufficiently to
 *   provide effective preconditioning.
 */
template <typename LinearSolverRegistrars =
              tmpl::list<Registrars::ExplicitInverse>>
class ExplicitInverse : public LinearSolver<LinearSolverRegistrars> {
 private:
  using Base = LinearSolver<LinearSolverRegistrars>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Build a matrix representation of the linear operator and invert it "
      "directly. This means that the first solve has a large initialization "
      "cost, but all subsequent solves converge immediately.";

  ExplicitInverse() = default;
  ExplicitInverse(const ExplicitInverse& /*rhs*/) = default;
  ExplicitInverse& operator=(const ExplicitInverse& /*rhs*/) = default;
  ExplicitInverse(ExplicitInverse&& /*rhs*/) = default;
  ExplicitInverse& operator=(ExplicitInverse&& /*rhs*/) = default;
  ~ExplicitInverse() = default;

  /// \cond
  explicit ExplicitInverse(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ExplicitInverse);  // NOLINT
  /// \endcond

  /*!
   * \brief Solve the equation \f$Ax=b\f$ by explicitly constructing the
   * operator matrix \f$A\f$ and its inverse. The first solve is computationally
   * expensive and successive solves are cheap.
   *
   * Building a matrix representation of the `linear_operator` requires
   * iterating over the `SourceType` (which is also the type returned by the
   * `linear_operator`) in a consistent way. This can be non-trivial for
   * heterogeneous data structures because it requires they define a data
   * ordering. Specifically, the `SourceType` must have a `size()` function as
   * well as `begin()` and `end()` iterators that point into the data. If the
   * iterators have a `reset()` function it is used to avoid repeatedly
   * re-creating the `begin()` iterator. The `reset()` function must not
   * invalidate the `end()` iterator.
   */
  template <typename LinearOperator, typename VarsType, typename SourceType,
            typename... OperatorArgs>
  Convergence::HasConverged solve(
      gsl::not_null<VarsType*> solution, const LinearOperator& linear_operator,
      const SourceType& source,
      const std::tuple<OperatorArgs...>& operator_args =
          std::tuple{}) const noexcept;

  /// Flags the operator to require re-initialization. No memory is released.
  /// Call this function to rebuild the solver when the operator changed.
  void reset() noexcept override {
    size_ = std::numeric_limits<size_t>::max();
  }

  /// Size of the operator. The stored matrix will have `size^2` entries.
  size_t size() const noexcept { return size_; }

  /// The matrix representation of the solver. This matrix approximates the
  /// inverse of the subdomain operator.
  const DenseMatrix<double>& matrix_representation() const noexcept {
    return inverse_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    p | size_;
    p | inverse_;
    if (p.isUnpacking() and size_ != std::numeric_limits<size_t>::max()) {
      source_workspace_.resize(size_);
      solution_workspace_.resize(size_);
    }
  }

  std::unique_ptr<Base> get_clone() const noexcept override {
    return std::make_unique<ExplicitInverse>(*this);
  }

 private:
  // Caches for successive solves of the same operator
  mutable size_t size_ = std::numeric_limits<size_t>::max();
  mutable DenseMatrix<double, blaze::columnMajor> inverse_{};

  // Buffers to avoid re-allocating memory for applying the operator
  mutable DenseVector<double> source_workspace_{};
  mutable DenseVector<double> solution_workspace_{};
};

template <typename LinearSolverRegistrars>
template <typename LinearOperator, typename VarsType, typename SourceType,
          typename... OperatorArgs>
Convergence::HasConverged ExplicitInverse<LinearSolverRegistrars>::solve(
    const gsl::not_null<VarsType*> solution,
    const LinearOperator& linear_operator, const SourceType& source,
    const std::tuple<OperatorArgs...>& operator_args) const noexcept {
  if (UNLIKELY(size_ == std::numeric_limits<size_t>::max())) {
    const auto& used_for_size = source;
    size_ = used_for_size.size();
    source_workspace_.resize(size_);
    solution_workspace_.resize(size_);
    inverse_.resize(size_, size_);
    // Construct explicit matrix representation by "sniffing out" the operator,
    // i.e. feeding it unit vectors
    auto unit_vector = make_with_value<VarsType>(used_for_size, 0.);
    auto result_buffer = make_with_value<SourceType>(used_for_size, 0.);
    auto& operator_matrix = inverse_;
    size_t i = 0;
    // Re-using the iterators for all operator invocations
    auto result_iterator_begin = result_buffer.begin();
    auto result_iterator_end = result_buffer.end();
    for (double& unit_vector_data : unit_vector) {
      // Add a 1 at the unit vector location i
      unit_vector_data = 1.;
      // Invoke the operator on the unit vector
      std::apply(linear_operator,
                 std::tuple_cat(std::forward_as_tuple(
                                    make_not_null(&result_buffer), unit_vector),
                                operator_args));
      // Set the unit vector back to zero
      unit_vector_data = 0.;
      // Reset the iterator by calling its `reset` member function or by
      // re-creating it
      if constexpr (detail::is_reset_callable_v<decltype(
                        result_iterator_begin)>) {
        result_iterator_begin.reset();
      } else {
        result_iterator_begin = result_buffer.begin();
        result_iterator_end = result_buffer.end();
      }
      // Store the result in column i of the matrix
      std::copy(result_iterator_begin, result_iterator_end,
                column(operator_matrix, i).begin());
      ++i;
    }
    // Directly invert the matrix
    try {
      blaze::invert(inverse_);
    } catch (const std::invalid_argument& e) {
      ERROR("Could not invert subdomain matrix (size " << size_
                                                       << "): " << e.what());
    }
  }
  // Copy source into contiguous workspace. In cases where the source and
  // solution data are already stored contiguously we might avoid the copy and
  // the associated workspace memory. However, compared to the cost of building
  // and storing the matrix this is likely insignificant.
  std::copy(source.begin(), source.end(), source_workspace_.begin());
  // Apply inverse
  solution_workspace_ = inverse_ * source_workspace_;
  // Reconstruct solution data from contiguous workspace
  std::copy(solution_workspace_.begin(), solution_workspace_.end(),
            solution->begin());
  return {0, 0};
}

/// \cond
template <typename LinearSolverRegistrars>
// NOLINTNEXTLINE
PUP::able::PUP_ID ExplicitInverse<LinearSolverRegistrars>::my_PUP_ID = 0;
/// \endcond

}  // namespace LinearSolver::Serial
