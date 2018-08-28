// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the linear solver

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"

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
 * \brief Holds an `IterationId` that identifies a step in the linear solver
 * algorithm
 */
struct IterationId : db::SimpleTag {
  static std::string name() noexcept { return "IterationId"; }
  using type = LinearSolver::IterationId;
};

/*!
 * \brief The operand that the local linear operator \f$A\f$ is applied to
 *
 * \details The result of the operation should be wrapped in
 * `LinearSolver::Tags::OperatorAppliedTo`.
 */
template <typename Tag>
struct Operand : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
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
    return "LinearOperatorAppliedTo(" + Tag::name() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief The residual \f$r=b - Ax\f$
 */
template <typename Tag>
struct Residual : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "LinearResidual(" + Tag::name() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief The magnitude square of the residual \f$\langle r,r\rangle\f$ w.r.t.
 * the `LinearSolver::inner_product`
 */
template <typename Tag>
struct ResidualMagnitudeSquare : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "LinearResidualMagnitudeSquare(" + Tag::name() + ")";
  }
  using type = double;
  using tag = Tag;
};

}  // namespace Tags
}  // namespace LinearSolver
