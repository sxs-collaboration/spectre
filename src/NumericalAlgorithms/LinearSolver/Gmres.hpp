// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <cstddef>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"

namespace LinearSolver {
namespace gmres_detail {

// Perform an Arnoldi orthogonalization to find a new `operand` that is
// orthogonal to all vectors in `basis_history`. Appends a new column to the
// `orthogonalization_history` that holds the inner product of the intermediate
// `operand` with each vector in the `basis_history` and itself.
template <typename VarsType>
void arnoldi_orthogonalize(
    const gsl::not_null<VarsType*> operand,
    const gsl::not_null<DenseMatrix<double>*> orthogonalization_history,
    const std::vector<VarsType>& basis_history,
    const size_t iteration) noexcept {
  // Resize matrix and make sure the new entries that are not being filled below
  // are zero.
  orthogonalization_history->resize(iteration + 2, iteration + 1);
  for (size_t j = 0; j < iteration; ++j) {
    (*orthogonalization_history)(iteration + 1, j) = 0.;
  }
  // Arnoldi orthogonalization
  for (size_t j = 0; j < iteration + 1; ++j) {
    const double orthogonalization = inner_product(basis_history[j], *operand);
    (*orthogonalization_history)(j, iteration) = orthogonalization;
    *operand -= orthogonalization * basis_history[j];
  }
  (*orthogonalization_history)(iteration + 1, iteration) =
      sqrt(inner_product(*operand, *operand));
  // Avoid an FPE if the new operand norm is exactly zero. In that case the
  // problem is solved and the algorithm will terminate (see Proposition 9.3 in
  // \cite Saad2003). Since there will be no next iteration we don't need to
  // normalize the operand.
  if (UNLIKELY((*orthogonalization_history)(iteration + 1, iteration) == 0.)) {
    return;
  }
  *operand /= (*orthogonalization_history)(iteration + 1, iteration);
}

// Solve the linear least-squares problem `||beta - H * y||` for `y`, where `H`
// is the Hessenberg matrix given by `orthogonalization_history` and `beta` is
// the vector `(initial_residual, 0, 0, ...)` by updating the QR decomposition
// of `H` from the previous iteration with a Givens rotation.
void solve_minimal_residual(
    gsl::not_null<DenseMatrix<double>*> orthogonalization_history,
    gsl::not_null<DenseVector<double>*> residual_history,
    gsl::not_null<DenseVector<double>*> givens_sine_history,
    gsl::not_null<DenseVector<double>*> givens_cosine_history,
    size_t iteration) noexcept;

// Find the vector that minimizes the residual by inverting the upper
// triangular matrix obtained above.
DenseVector<double> minimal_residual_vector(
    const DenseMatrix<double>& orthogonalization_history,
    const DenseVector<double>& residual_history) noexcept;

}  // namespace gmres_detail

namespace Serial {

template <typename VarsType>
struct IdentityPreconditioner {
  VarsType operator()(const VarsType& arg) const noexcept { return arg; }
};

struct NoIterationCallback {
  void operator()(const Convergence::HasConverged& /*has_converged*/) const
      noexcept {}
};

/*!
 * \brief A serial GMRES iterative solver for nonsymmetric linear systems of
 * equations.
 *
 * This is an iterative algorithm to solve general linear equations \f$Ax=b\f$
 * where \f$A\f$ is a linear operator. See \cite Saad2003, chapter 6.5 for a
 * description of the GMRES algorithm and Algorithm 9.6 for this implementation.
 * It is matrix-free, which means the operator \f$A\f$ needs not be provided
 * explicity as a matrix but only the operator action \f$A(x)\f$ must be
 * provided for an argument \f$x\f$.
 *
 * The GMRES algorithm does not require the operator \f$A\f$ to be symmetric or
 * positive-definite. Note that other algorithms such as conjugate gradients may
 * be more efficient for symmetric positive-definite operators.
 *
 * \par Convergence:
 * Given a set of \f$N_A\f$ equations (e.g. through an \f$N_A\times N_A\f$
 * matrix) the GMRES algorithm will converge to numerical precision in at most
 * \f$N_A\f$ iterations. However, depending on the properties of the linear
 * operator, an approximate solution can ideally be obtained in only a few
 * iterations. See \cite Saad2003, section 6.11.4 for details on the convergence
 * of the GMRES algorithm.
 *
 * \par Restarting:
 * This implementation of the GMRES algorithm supports restarting, as detailed
 * in \cite Saad2003, section 6.5.5. Since the GMRES algorithm iteratively
 * builds up an orthogonal basis of the solution space the cost of each
 * iteration increases linearly with the number of iterations. Therefore it is
 * sometimes helpful to restart the algorithm every \f$N_\mathrm{restart}\f$
 * iterations, discarding the set of basis vectors and starting again from the
 * current solution estimate. This strategy can improve the performance of the
 * solver, but note that the solver can stagnate for non-positive-definite
 * operators and is not guaranteed to converge within \f$N_A\f$ iterations
 * anymore. Set the `restart` argument of the constructor to
 * \f$N_\mathrm{restart}\f$ to activate restarting, or set it to zero to
 * deactivate restarting (default behaviour).
 *
 * \par Preconditioning:
 * This implementation of the GMRES algorithm also supports preconditioning.
 * You can provide a linear operator \f$P\f$ that approximates the inverse of
 * the operator \f$A\f$ to accelerate the convergence of the linear solve.
 * The algorithm is right-preconditioned, which allows the preconditioner to
 * change in every iteration ("flexible" variant). See \cite Saad2003, sections
 * 9.3.2 and 9.4.1 for details. This implementation follows Algorithm 9.6 in
 * \cite Saad2003.
 *
 * \par Improvements:
 * Further improvements can potentially be implemented for this algorithm, see
 * e.g. \cite Ayachour2003.
 *
 * \example
 * \snippet NumericalAlgorithms/LinearSolver/Test_Gmres.cpp gmres_example
 */
template <typename VarsType>
struct Gmres {
 private:
  struct ConvergenceCriteria {
    using type = Convergence::Criteria;
    static constexpr OptionString help =
        "Determine convergence of the algorithm";
  };
  struct Restart {
    using type = size_t;
    static constexpr OptionString help = "Iterations to run before restarting";
    static size_t default_value() noexcept { return 0; }
  };
  struct Verbosity {
    using type = ::Verbosity;
    static constexpr OptionString help = "Logging verbosity";
  };

 public:
  static constexpr OptionString help =
      "A serial GMRES iterative solver for nonsymmetric linear systems of\n"
      "equations Ax=b. It will converge to numerical precision in at most N_A\n"
      "iterations, where N_A is the number of equations represented by the\n"
      "linear operator A, but will ideally converge to a reasonable\n"
      "approximation of the solution x in only a few iterations.\n"
      "\n"
      "Restarting: It is sometimes helpful to restart the algorithm every\n"
      "N_restart iterations to speed it up. Note that it can stagnate for\n"
      "non-positive-definite matrices and is not guaranteed to converge\n"
      "within N_A iterations anymore when restarting is activated.\n"
      "Activate restarting by setting the 'Restart' option to N_restart, or\n"
      "deactivate restarting by setting it to zero (default).";
  using options = tmpl::list<ConvergenceCriteria, Verbosity, Restart>;

  Gmres(Convergence::Criteria convergence_criteria, ::Verbosity verbosity,
        size_t restart = 0) noexcept
      // clang-tidy: trivially copyable
      : convergence_criteria_(std::move(convergence_criteria)),  // NOLINT
        verbosity_(std::move(verbosity)),                        // NOLINT
        restart_(restart > 0 ? restart : convergence_criteria_.max_iterations) {
    initialize();
  }

  Gmres() = default;
  Gmres(const Gmres& /*rhs*/) = default;
  Gmres& operator=(const Gmres& /*rhs*/) = default;
  Gmres(Gmres&& /*rhs*/) = default;
  Gmres& operator=(Gmres&& /*rhs*/) = default;
  ~Gmres() = default;

  void initialize() noexcept {
    orthogonalization_history_.reserve(restart_ + 1);
    residual_history_.reserve(restart_ + 1);
    givens_sine_history_.reserve(restart_);
    givens_cosine_history_.reserve(restart_);
    basis_history_.resize(restart_ + 1);
    preconditioned_basis_history_.resize(restart_);
  }

  const Convergence::Criteria& convergence_criteria() const noexcept {
    return convergence_criteria_;
  }

  void pup(PUP::er& p) noexcept {  // NOLINT
    p | convergence_criteria_;
    p | verbosity_;
    p | restart_;
    if (p.isUnpacking()) {
      initialize();
    }
  }

  /*!
   * \brief Iteratively solve the problem \f$Ax=b\f$ for \f$x\f$ where \f$A\f$
   * is the `linear_operator` and \f$b\f$ is the `source`, starting \f$x\f$ at
   * `initial_guess`.
   *
   * Optionally provide a `preconditioner` (see class documentation).
   *
   * \return An instance of `Convergence::HasConverged` that provides
   * information on the convergence status of the completed solve, and the
   * approximate solution \f$x\f$.
   */
  template <typename LinearOperator, typename SourceType,
            typename Preconditioner = IdentityPreconditioner<VarsType>,
            typename IterationCallback = NoIterationCallback>
  std::pair<Convergence::HasConverged, VarsType> operator()(
      LinearOperator&& linear_operator, const SourceType& source,
      const VarsType& initial_guess,
      Preconditioner&& preconditioner = IdentityPreconditioner<VarsType>{},
      IterationCallback&& = NoIterationCallback{}) const noexcept;

 private:
  Convergence::Criteria convergence_criteria_{};
  ::Verbosity verbosity_{::Verbosity::Verbose};
  size_t restart_{};

  // Memory buffers to avoid re-allocating memory for successive solves:
  // The `orthogonalization_history_` is built iteratively from inner products
  // between existing and potential basis vectors and then Givens-rotated to
  // become upper-triangular.
  mutable DenseMatrix<double> orthogonalization_history_{};
  // The `residual_history_` holds the remaining residual in its last entry, and
  // the other entries `g` "source" the minimum residual vector `y` in
  // `R * y = g` where `R` is the upper-triangular `orthogonalization_history_`.
  mutable DenseVector<double> residual_history_{};
  // These represent the accumulated Givens rotations up to the current
  // iteration.
  mutable DenseVector<double> givens_sine_history_{};
  mutable DenseVector<double> givens_cosine_history_{};
  // These represent the orthogonal Krylov-subspace basis that is constructed
  // iteratively by Arnoldi-orthogonalizing a new vector in each iteration and
  // appending it to the `basis_history_`.
  mutable std::vector<VarsType> basis_history_{};
  // When a preconditioner is used it is applied to each new basis vector. The
  // preconditioned basis is used to construct the solution when the algorithm
  // has converged.
  mutable std::vector<VarsType> preconditioned_basis_history_{};
};

template <typename VarsType>
template <typename LinearOperator, typename SourceType, typename Preconditioner,
          typename IterationCallback>
std::pair<Convergence::HasConverged, VarsType> Gmres<VarsType>::operator()(
    LinearOperator&& linear_operator, const SourceType& source,
    const VarsType& initial_guess, Preconditioner&& preconditioner,
    IterationCallback&& iteration_callback) const noexcept {
  constexpr bool use_preconditioner =
      not cpp17::is_same_v<Preconditioner, IdentityPreconditioner<VarsType>>;
  constexpr bool use_iteration_callback =
      not cpp17::is_same_v<IterationCallback, NoIterationCallback>;

  auto result = initial_guess;
  Convergence::HasConverged has_converged{};
  size_t iteration = 0;

  while (not has_converged) {
    auto& initial_operand = basis_history_[0] =
        source - linear_operator(result);
    const double initial_residual_magnitude =
        sqrt(inner_product(initial_operand, initial_operand));
    has_converged = Convergence::HasConverged{convergence_criteria_, iteration,
                                              initial_residual_magnitude,
                                              initial_residual_magnitude};
    if (use_iteration_callback) {
      iteration_callback(has_converged);
    }
    if (UNLIKELY(has_converged)) {
      break;
    }
    initial_operand /= initial_residual_magnitude;
    residual_history_.resize(1);
    residual_history_[0] = initial_residual_magnitude;
    for (size_t k = 0; k < restart_; ++k) {
      auto& operand = basis_history_[k + 1];
      if (use_preconditioner) {
        preconditioned_basis_history_[k] = preconditioner(basis_history_[k]);
      }
      operand =
          linear_operator(use_preconditioner ? preconditioned_basis_history_[k]
                                             : basis_history_[k]);
      // Find a new orthogonal basis vector of the Krylov subspace
      gmres_detail::arnoldi_orthogonalize(
          make_not_null(&operand), make_not_null(&orthogonalization_history_),
          basis_history_, k);
      // Least-squares solve for the minimal residual
      gmres_detail::solve_minimal_residual(
          make_not_null(&orthogonalization_history_),
          make_not_null(&residual_history_),
          make_not_null(&givens_sine_history_),
          make_not_null(&givens_cosine_history_), k);
      ++iteration;
      has_converged = Convergence::HasConverged{
          convergence_criteria_, iteration, abs(residual_history_[k + 1]),
          initial_residual_magnitude};
      if (use_iteration_callback) {
        iteration_callback(has_converged);
      }
      if (UNLIKELY(has_converged)) {
        break;
      }
    }
    // Find the vector w.r.t. the constructed orthogonal basis of the Krylov
    // subspace that minimizes the residual
    const auto minres = gmres_detail::minimal_residual_vector(
        orthogonalization_history_, residual_history_);
    // Construct the solution from the orthogonal basis and the minimal residual
    // vector
    for (size_t i = 0; i < minres.size(); ++i) {
      result +=
          minres[i] * gsl::at(use_preconditioner ? preconditioned_basis_history_
                                                 : basis_history_,
                              i);
    }
  }
  return {std::move(has_converged), std::move(result)};
}

}  // namespace Serial
}  // namespace LinearSolver
