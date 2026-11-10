// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function find_real_eigensystem.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Find the eigensystem of a diagonalizable matrix whose eigenvalues are
 * all real, with a complete set of eigenvectors even when degeneracies are
 * present.
 *
 * We label the matrix $A_{m}{}^{n}$ such that the index $m$ labels rows and the
 * index $n$ labels columns.
 *
 * With this choice of indices, the $i$th left eigensystem is given by
 * \begin{equation}
 *   \sum_m L^{im} A_{m}{}^{n} = \lambda_{(i)} L^{in},
 * \end{equation}
 * where $\lambda_{(i)}$ is the $i$th eigenvalue.
 * Similarly, the $i$th right eigensystem is given by
 * \begin{equation}
 *   \sum_n R_{in} A_{m}{}^{n} = \lambda_{(i)} R_{im}.
 * \end{equation}
 *
 * We take the matrix and provide the eigensystem in the following types:
 * - %Matrix $A_{m}{}^{n}$: `tnsr::iJ<DataVector, MatrixSize> matrix`
 * - Eigenvalues $\lambda_{(i)}$: `tnsr::i<DataVector, MatrixSize> eigenvalues`
 * - Right eigenvectors $R_{in}$: `tnsr::ij<DataVector, MatrixSize>
 *   right_eigenvectors`
 * - Left eigenvectors $L^{im}$: `tnsr::IJ<DataVector, MatrixSize>
 *   left_eigenvectors`
 *
 * Note that the indices $i$, $m$, and $n$ are not tensorial. They simply label
 * the eigensystem, rows of the matrix, and columns of the matrix, respectively.
 * We make the conventions (raised or lowered) presented above to clearly
 * distinguish which indices (first or second) of the matrix get contracted
 * with which eigenvectors (left or right).
 *
 * Each grid point holds an independent matrix. Repeated (degenerate)
 * eigenvalues are allowed, as long as the matrix is diagonalizable. The
 * eigenvalues are in increasing order, with degenerate eigenvalues grouped
 * together.
 *
 * To compute such an eigensystem, we follow the procedure outlined below, each
 * step using a LAPACK routine.
 *
 * 1. Balance the matrix (`dgebal`), $A' = D^{-1} A D$, where $D$ is a diagonal
 *    matrix chosen so that the rows and columns of $A'$ have similar norms.
 *    This improves the accuracy of the next steps and does not change the
 *    eigenvalues.
 * 2. Compute the real Schur form of the balanced matrix (`dgees`),
 *    $A' = Z T Z^T$, where $Z$ is orthogonal and $T$ is upper triangular with
 *    the eigenvalues on its diagonal. The first column of $Z$ is a right
 *    eigenvector of $A'$ for the first eigenvalue on the diagonal of $T$. In
 *    general, the other columns are not eigenvectors. The exception is a
 *    repeated eigenvalue (i.e., when we have a degeneracy): if the first $k$
 *    eigenvalues on the diagonal of $T$ are equal, the first $k$ columns of $Z$
 *    are $k$ orthonormal right eigenvectors for that degenerate subspace.
 * 3. Group the eigenvalues into degenerate families, i.e., sets of equal
 *    eigenvalues. For each family, of size $k$, reorder the Schur form
 *    (`dtrsen`), $T \to Q^T T Q$ and $Z \to Z Q$ with $Q$ orthogonal, so that
 *    the family's eigenvalues come first on the diagonal of $T$, and take the
 *    first $k$ columns of $Z$ as the family's right eigenvectors. The
 *    reordering rotates the columns of $Z$, keeping them orthonormal and $T$
 *    upper triangular.
 * 4. Compute the left eigenvectors as the inverse of the right eigenvectors
 *    (`dgesv`), so that $\sum_m L^{im} R_{jm} = \delta^{i}_{j}$.
 *
 * Finally, we undo the balancing: $R_{im} = D_{mm} R'_{im}$ and
 * $L^{im} = L'^{im} / D_{mm}$.
 *
 * Eigenvalues are treated as equal (degenerate) when they differ by at most
 * $10^{-10} \|A'\|_F$, where $\|A'\|_F$ is the Frobenius norm of the
 * balanced matrix.
 */
template <size_t MatrixSize>
void find_real_eigensystem(
    gsl::not_null<tnsr::i<DataVector, MatrixSize>*> eigenvalues,
    gsl::not_null<tnsr::ij<DataVector, MatrixSize>*> right_eigenvectors,
    gsl::not_null<tnsr::IJ<DataVector, MatrixSize>*> left_eigenvectors,
    const tnsr::iJ<DataVector, MatrixSize>& matrix);
