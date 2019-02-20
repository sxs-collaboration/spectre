// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the linear solver

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "Informer/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Convergence.hpp"
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
#include "Options/Options.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/*!
 * \ingroup LinearSolverGroup
 * \brief Functionality for solving linear systems of equations
 */
namespace LinearSolver {

/*!
 * \ingroup LinearSolverGroup
 * \brief The \ref DataBoxGroup tags associated with the linear solver
 */
namespace Tags {

/*!
 * \brief The operand that the local linear operator \f$A\f$ is applied to
 *
 * \details The result of the operation should be wrapped in
 * `LinearSolver::Tags::OperatorAppliedTo`.
 */
template <typename Tag>
struct Operand : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearOperand(" + Tag::name() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief The linear operator \f$A\f$ applied to the data in `Tag`
 */
template <typename Tag>
struct OperatorAppliedTo : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearOperatorAppliedTo(" + Tag::name() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief Holds an `IterationId` that identifies a step in the linear solver
 * algorithm
 */
struct IterationId : db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearIterationId";
  }
  using type = LinearSolver::IterationId;
  template <typename Tag>
  using step_prefix = OperatorAppliedTo<Tag>;
};

/*!
 * \brief The residual \f$r=b - Ax\f$
 */
template <typename Tag>
struct Residual : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearResidual(" + Tag::name() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

template <typename Tag>
struct Initial : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "Initial(" + Tag::name() + ")"; }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief The magnitude square \f$\langle \cdot,\cdot\rangle\f$ w.r.t.
 * the `LinearSolver::inner_product`
 */
template <typename Tag>
struct MagnitudeSquare : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearMagnitudeSquare(" + Tag::name() + ")";
  }
  using type = double;
  using tag = Tag;
};

/*!
 * \brief The magnitude \f$\sqrt{\langle \cdot,\cdot\rangle}\f$ w.r.t.
 * the `LinearSolver::inner_product`
 */
template <typename Tag>
struct Magnitude : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearMagnitude(" + Tag::name() + ")";
  }
  using type = double;
  using tag = Tag;
};

/*!
 * \brief Compute the `LinearSolver::Magnitude` of a tag from its
 * `LinearSolver::MagnitudeSquare`.
 */
template <typename MagnitudeSquareTag,
          Requires<tt::is_a_v<MagnitudeSquare, MagnitudeSquareTag>> = nullptr>
struct MagnitudeCompute
    : db::add_tag_prefix<Magnitude, db::remove_tag_prefix<MagnitudeSquareTag>>,
      db::ComputeTag {
  static constexpr double function(const double& magnitude_square) noexcept {
    return sqrt(magnitude_square);
  }
  using argument_tags = tmpl::list<MagnitudeSquareTag>;
};

/*!
 * \brief The prefix for tags related to an orthogonalization procedurce
 */
template <typename Tag>
struct Orthogonalization : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearOrthogonalization(" + Tag::name() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief A Hessenberg matrix built up during an orthogonalization procedure
 */
template <typename Tag>
struct OrthogonalizationHistory : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearOrthogonalizationHistory(" + Tag::name() + ")";
  }
  using type = DenseMatrix<double>;
  using tag = Tag;
};

/*!
 * \brief A set of \f$n\f$ vectors that form a basis of the \f$n\f$-th Krylov
 * subspace \f$K_n(A,b)\f$
 *
 * \details The Krylov subspace \f$K_n(A,b)\f$ spanned by this basis is the one
 * generated by the linear operator \f$A\f$ and source \f$b\f$ that are
 * represented by the tags
 * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 * db::add_tag_prefix<LinearSolver::Tags::Operand, Tag>>` and
 * `db::add_tag_prefix<::Tags::Source, Tag>`, respectively. Therefore, each
 * basis vector is of the type `db::item_type<db::add_tag_prefix<Operand,
 * Tag>>`.
 */
template <typename Tag>
struct KrylovSubspaceBasis : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // No "Linear" prefix since a Krylov subspace always refers to a linear
    // operator
    return "KrylovSubspaceBasis(" + Tag::name() + ")";
  }
  using type = std::vector<db::item_type<db::add_tag_prefix<Operand, Tag>>>;
  using tag = Tag;
};

/*!
 * \brief `LinearSolver::ConvergenceCriteria` that determine the linear solve
 * has converged
 */
struct ConvergenceCriteria : db::SimpleTag {
  static std::string name() noexcept { return "ConvergenceCriteria"; }
  static constexpr OptionString help =
      "Criteria that determine the linear solve has converged";
  using type = LinearSolver::ConvergenceCriteria;
};

/*!
 * \brief Holds a `LinearSolver::HasConverged` flag that signals the linear
 * solver has converged, along with the reason for convergence.
 */
struct HasConverged : db::SimpleTag {
  static std::string name() noexcept { return "LinearSolverHasConverged"; }
  using type = LinearSolver::HasConverged;
};

/*
 * \brief Employs the `LinearSolver::Tags::ConvergenceCriteria` to determine the
 * linear solver has converged.
 */
template <typename FieldsTag>
struct HasConvergedCompute : LinearSolver::Tags::HasConverged, db::ComputeTag {
 private:
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>>;
  using initial_residual_magnitude_tag =
      db::add_tag_prefix<LinearSolver::Tags::Initial, residual_magnitude_tag>;

 public:
  using argument_tags =
      tmpl::list<LinearSolver::Tags::ConvergenceCriteria,
                 LinearSolver::Tags::IterationId, residual_magnitude_tag,
                 initial_residual_magnitude_tag>;
  static db::item_type<LinearSolver::Tags::HasConverged> function(
      const LinearSolver::ConvergenceCriteria& convergence_criteria,
      const LinearSolver::IterationId& iteration_id,
      const double& residual_magnitude,
      const double& initial_residual_magnitude) noexcept {
    return LinearSolver::HasConverged(convergence_criteria, iteration_id,
                                      residual_magnitude,
                                      initial_residual_magnitude);
  }
};

}  // namespace Tags

namespace OptionTags {

struct ResidualMonitorOptions
    : tuples::TaggedTuple<::OptionTags::Verbosity,
                          LinearSolver::Tags::ConvergenceCriteria> {
  // TypedTag and OptionTag interface
  static std::string name() noexcept { return "LinearSolver"; }
  using type = ResidualMonitorOptions;
  // OptionCreatable interface
  static constexpr OptionString help =
      "Options to control the linear solver algorithm.";
  using options = TaggedTuple::tags;
  using TaggedTuple::TaggedTuple;
};

}  // namespace OptionTags

}  // namespace LinearSolver
