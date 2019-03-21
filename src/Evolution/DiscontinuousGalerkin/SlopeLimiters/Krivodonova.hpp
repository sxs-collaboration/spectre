// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cmath>
#include <cstddef>
#include <limits>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <type_traits>
#include <unordered_map>
#include <utility>  // for pair

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"         // IWYU pragma: keep
#include "DataStructures/Variables.hpp"             // IWYU pragma: keep
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"    // IWYU pragma: keep
#include "Domain/ElementId.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"            // IWYU pragma: keep
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <algorithm>
// IWYU pragma: no_forward_declare Variables

/// \cond
namespace SlopeLimiters {
template <size_t VolumeDim, typename TagsToLimit>
class Krivodonova;
}  // namespace SlopeLimiters
/// \endcond

namespace SlopeLimiters {
/*!
 * \ingroup SlopeLimitersGroup
 * \brief An implementation of the Krivodonova slope limiter.
 *
 * The slope limiter is described in \cite Krivodonova2007. The Krivodonova
 * limiter works by limiting the highest derivatives/modal coefficients using an
 * aggressive minmod approach, decreasing in derivative/modal coefficient order
 * until no more limiting is necessary. In 3d, the function being limited is
 * expanded as:
 *
 * \f{align}{
 * u^{l,m,n}=\sum_{i,j,k=0,0,0}^{N_i,N_j,N_k}c^{l,m,n}_{i,j,k}
 *  P_{i}(\xi)P_{j}(\eta)P_{k}(\zeta)
 * \f}
 *
 * where \f$\left\{\xi, \eta, \zeta\right\}\f$ are the logical coordinates,
 * \f$P_{i}\f$ are the Legendre polynomials, the superscript \f$\{l,m,n\}\f$
 * represents the element indexed by \f$l,m,n\f$, and \f$N_i,N_j\f$ and
 * \f$N_k\f$ are the number of collocation points minus one in the
 * \f$\xi,\eta,\f$ and \f$\zeta\f$ direction, respectively. The coefficients are
 * limited according to:
 *
 * \f{align}{
 * \tilde{c}^{l,m,n}_{i,j,k}=\mathrm{minmod}
 *   &\left(c_{i,j,k}^{l,m,n},
 *          \alpha_i\left(c^{l+1,m,n}_{i-1,j,k}-c^{l,m,n}_{i-1,j,k}\right),
 *          \alpha_i\left(c^{l,m,n}_{i-1,j,k}-c^{l-1,m,n}_{i-1,j,k}\right),
 *     \right.\notag \\
 *   &\;\;\;\;
 *          \alpha_j\left(c^{l,m+1,n}_{i,j-1,k}-c^{l,m,n}_{i,j-1,k}\right),
 *          \alpha_j\left(c^{l,m,n}_{i,j-1,k}-c^{l,m-1,n}_{i,j-1,k}\right),
 *     \notag \\
 *   &\;\;\;\;\left.
 *          \alpha_k\left(c^{l,m,n+1}_{i,j,k-1}-c^{l,m,n}_{i,j,k-1}\right),
 *          \alpha_k\left(c^{l,m,n}_{i,j,k-1}-c^{l,m,n-1}_{i,j,k-1}\right)
 *     \right),
 * \label{eq:krivodonova 3d minmod}
 * \f}
 *
 * where \f$\mathrm{minmod}\f$ is the minmod function defined as
 *
 * \f{align}{
 *  \mathrm{minmod}(a,b,c,\ldots)=
 *  \left\{
 *  \begin{array}{ll}
 *    \mathrm{sgn}(a)\min(\lvert a\rvert, \lvert b\rvert,
 *    \lvert c\rvert, \ldots) & \mathrm{if} \;
 *    \mathrm{sgn}(a)=\mathrm{sgn}(b)=\mathrm{sgn}(c)=\mathrm{sgn}(\ldots) \\
 *    0 & \mathrm{otherwise}
 *  \end{array}\right.
 * \f}
 *
 * Krivodonova \cite Krivodonova2007 requires \f$\alpha_i\f$ to be in the range
 *
 * \f{align*}{
 * \frac{1}{2(2i-1)}\le \alpha_i \le 1
 * \f}
 *
 * where the lower bound comes from finite differencing the coefficients between
 * neighbor elements when using Legendre polynomials (see \cite Krivodonova2007
 * for details). Note that we normalize our Legendre polynomials by \f$P_i(1) =
 * 1\f$; this is the normalization \cite Krivodonova2007 uses in 1D, (but not in
 * 2D), which is why our bounds on \f$\alpha_i\f$ match Eq. 14 of
 * \cite Krivodonova2007 (but not Eq. 23). We relax the lower bound:
 *
 * \f{align*}{
 * 0 \le \alpha_i \le 1
 * \f}
 *
 * to allow different basis functions (e.g. Chebyshev polynomials) and to allow
 * the limiter to be more dissipative if necessary. The same \f$\alpha_i\f$s are
 * used in all dimensions.
 *
 * \note The only place where the specific choice of 1d basis
 * comes in is the lower bound for the \f$\alpha_i\f$s, and so in general the
 * limiter can be applied to any 1d or tensor product of 1d basis functions.
 *
 * The limiting procedure must be applied from the highest derivatives to the
 * lowest, i.e. the highest coefficients to the lowest. Let us consider a 3d
 * element with \f$N+1\f$ coefficients in each dimension and denote the
 * coefficients as \f$c_{i,j,k}\f$. Then the limiting procedure starts at
 * \f$c_{N,N,N}\f$, followed by \f$c_{N,N,N-1}\f$, \f$c_{N,N-1,N}\f$, and
 * \f$c_{N-1,N,N}\f$. A detailed example is given below. Limiting is stopped if
 * all symmetric pairs of coefficients are left unchanged, i.e.
 * \f$c_{i,j,k}=\tilde{c}_{i,j,k}\f$. By all symmetric coefficients we mean
 * that, for example, \f$c_{N-i,N-j,N-k}\f$, \f$c_{N-j,N-i,N-k}\f$,
 * \f$c_{N-k,N-j,N-i}\f$, \f$c_{N-j,N-k,N-i}\f$, \f$c_{N-i,N-k,N-j}\f$, and
 * \f$c_{N-k,N-i,N-j}\f$ are not limited. As a concrete example, consider a 3d
 * element with 3 collocation points per dimension. Each limited coefficient is
 * defined as (though only computed if needed):
 *
 * \f{align*}{
 * \tilde{c}^{l,m,n}_{2,2,2}=\mathrm{minmod}
 *   &\left(c_{2,2,2}^{l,m,n},
 *          \alpha_2\left(c^{l+1,m,n}_{1,2,2}-c^{l,m,n}_{1,2,2}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,2,2}-c^{l-1,m,n}_{1,2,2}\right),
 *     \right.\\
 *   &\;\;\;\;
 *          \alpha_2\left(c^{l,m+1,n}_{2,1,2}-c^{l,m,n}_{2,1,2}\right),
 *          \alpha_2\left(c^{l,m,n}_{2,1,2}-c^{l,m-1,n}_{2,1,2}\right),\\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m,n+1}_{2,2,1}-c^{l,m,n}_{2,2,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{2,2,1}-c^{l,m,n-1}_{2,2,1}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{2,2,1}=\mathrm{minmod}
 *   &\left(c_{2,2,1}^{l,m,n},
 *          \alpha_2\left(c^{l+1,m,n}_{1,2,1}-c^{l,m,n}_{1,2,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,2,1}-c^{l-1,m,n}_{1,2,1}\right),
 *     \right.\\
 *   &\;\;\;\;
 *          \alpha_2\left(c^{l,m+1,n}_{2,1,1}-c^{l,m,n}_{2,1,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{2,1,1}-c^{l,m-1,n}_{2,1,1}\right),\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m,n+1}_{2,2,0}-c^{l,m,n}_{2,2,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{2,2,0}-c^{l,m,n-1}_{2,2,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{2,1,2}=\mathrm{minmod}
 *   &\left(c_{2,1,2}^{l,m,n},
 *          \alpha_2\left(c^{l+1,m,n}_{1,1,2}-c^{l,m,n}_{1,1,2}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,1,2}-c^{l-1,m,n}_{1,1,2}\right),
 *     \right.\\
 *   &\;\;\;\;
 *          \alpha_1\left(c^{l,m+1,n}_{2,0,2}-c^{l,m,n}_{2,0,2}\right),
 *          \alpha_1\left(c^{l,m,n}_{2,0,2}-c^{l,m-1,n}_{2,0,2}\right),\\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m,n+1}_{2,1,1}-c^{l,m,n}_{2,1,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{2,1,1}-c^{l,m,n-1}_{2,1,1}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{1,2,2}=\mathrm{minmod}
 *   &\left(c_{1,2,2}^{l,m,n},
 *          \alpha_1\left(c^{l+1,m,n}_{0,2,2}-c^{l,m,n}_{0,2,2}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,2,2}-c^{l-1,m,n}_{0,2,2}\right),
 *     \right.\\
 *   &\;\;\;\;
 *          \alpha_2\left(c^{l,m+1,n}_{1,1,2}-c^{l,m,n}_{1,1,2}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,1,2}-c^{l,m-1,n}_{1,1,2}\right),\\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m,n+1}_{1,2,1}-c^{l,m,n}_{1,2,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,2,1}-c^{l,m,n-1}_{1,2,1}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{2,2,0}=\mathrm{minmod}
 *   &\left(c_{2,2,0}^{l,m,n},
 *          \alpha_2\left(c^{l+1,m,n}_{1,2,0}-c^{l,m,n}_{1,2,0}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,2,0}-c^{l-1,m,n}_{1,2,0}\right),
 *     \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m+1,n}_{2,1,0}-c^{l,m,n}_{2,1,0}\right),
 *          \alpha_2\left(c^{l,m,n}_{2,1,0}-c^{l,m-1,n}_{2,1,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{2,0,2}=\mathrm{minmod}
 *   &\left(c_{2,0,2}^{l,m,n},
 *          \alpha_2\left(c^{l+1,m,n}_{1,0,2}-c^{l,m,n}_{1,0,2}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,0,2}-c^{l-1,m,n}_{1,0,2}\right),
 *     \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m,n+1}_{2,0,1}-c^{l,m,n}_{2,0,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{2,0,1}-c^{l,m,n-1}_{2,0,1}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{0,2,2}=\mathrm{minmod}
 *   &\left(c_{0,2,2}^{l,m,n},
 *          \alpha_2\left(c^{l,m+1,n}_{0,1,2}-c^{l,m,n}_{0,1,2}\right),
 *          \alpha_2\left(c^{l,m,n}_{0,1,2}-c^{l,m-1,n}_{0,1,2}\right),
 *     \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m,n+1}_{0,2,1}-c^{l,m,n}_{0,2,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{0,2,1}-c^{l,m,n-1}_{0,2,1}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{2,1,1}=\mathrm{minmod}
 *   &\left(c_{2,1,1}^{l,m,n},
 *          \alpha_2\left(c^{l+1,m,n}_{1,1,1}-c^{l,m,n}_{1,1,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,1,1}-c^{l-1,m,n}_{1,1,1}\right),
 *     \right.\\
 *   &\;\;\;\;
 *          \alpha_1\left(c^{l,m+1,n}_{2,0,1}-c^{l,m,n}_{2,0,1}\right),
 *          \alpha_1\left(c^{l,m,n}_{2,0,1}-c^{l,m-1,n}_{2,0,1}\right),\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m,n+1}_{2,1,0}-c^{l,m,n}_{2,1,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{2,1,0}-c^{l,m,n-1}_{2,1,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{1,2,1}=\mathrm{minmod}
 *   &\left(c_{1,2,1}^{l,m,n},
 *          \alpha_1\left(c^{l+1,m,n}_{0,2,1}-c^{l,m,n}_{0,2,1}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,2,1}-c^{l-1,m,n}_{0,2,1}\right),
 *     \right.\\
 *   &\;\;\;\;
 *          \alpha_2\left(c^{l,m+1,n}_{1,1,1}-c^{l,m,n}_{1,1,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,1,1}-c^{l,m-1,n}_{1,1,1}\right),\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m,n+1}_{1,2,0}-c^{l,m,n}_{1,2,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{1,2,0}-c^{l,m,n-1}_{1,2,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{1,1,2}=\mathrm{minmod}
 *   &\left(c_{1,1,2}^{l,m,n},
 *          \alpha_1\left(c^{l+1,m,n}_{0,1,2}-c^{l,m,n}_{0,1,2}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,1,2}-c^{l-1,m,n}_{0,1,2}\right),
 *     \right.\\
 *   &\;\;\;\;
 *          \alpha_1\left(c^{l,m+1,n}_{1,0,2}-c^{l,m,n}_{1,0,2}\right),
 *          \alpha_1\left(c^{l,m,n}_{1,0,2}-c^{l,m-1,n}_{1,0,2}\right),\\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m,n+1}_{1,1,1}-c^{l,m,n}_{1,1,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,1,1}-c^{l,m,n-1}_{1,1,1}\right)
 *     \right),
 * \f}
 * \f{align*}{
 * \tilde{c}^{l,m,n}_{2,1,0}=\mathrm{minmod}
 *   &\left(c_{2,1,0}^{l,m,n},
 *          \alpha_2\left(c^{l+1,m,n}_{1,1,0}-c^{l,m,n}_{1,1,0}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,1,0}-c^{l-1,m,n}_{1,1,0}\right),
 *     \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m+1,n}_{2,0,0}-c^{l,m,n}_{2,0,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{2,0,0}-c^{l,m-1,n}_{2,0,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{2,0,1}=\mathrm{minmod}
 *   &\left(c_{2,0,1}^{l,m,n},
 *          \alpha_2\left(c^{l+1,m,n}_{1,0,1}-c^{l,m,n}_{1,0,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,0,1}-c^{l-1,m,n}_{1,0,1}\right),
 *     \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m,n+1}_{2,0,0}-c^{l,m,n}_{2,0,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{2,0,0}-c^{l,m,n-1}_{2,0,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{1,2,0}=\mathrm{minmod}
 *   &\left(c_{1,2,0}^{l,m,n},
 *          \alpha_1\left(c^{l+1,m,n}_{0,2,0}-c^{l,m,n}_{0,2,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,2,0}-c^{l-1,m,n}_{0,2,0}\right),
 *     \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m+1,n}_{1,1,0}-c^{l,m,n}_{1,1,0}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,1,0}-c^{l,m-1,n}_{1,1,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{1,0,2}=\mathrm{minmod}
 *   &\left(c_{1,0,2}^{l,m,n},
 *          \alpha_1\left(c^{l+1,m,n}_{0,0,2}-c^{l,m,n}_{0,0,2}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,0,2}-c^{l-1,m,n}_{0,0,2}\right),
 *     \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m,n+1}_{1,0,1}-c^{l,m,n}_{1,0,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,0,1}-c^{l,m,n-1}_{1,0,1}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{0,1,2}=\mathrm{minmod}
 *   &\left(c_{0,1,2}^{l,m,n},
 *          \alpha_1\left(c^{l,m+1,n}_{0,0,2}-c^{l,m,n}_{0,0,2}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,0,2}-c^{l,m-1,n}_{0,0,2}\right),
 *   \right. \\
 *   &\;\;\;\;\left.
 *          \alpha_2\left(c^{l,m,n+1}_{0,1,1}-c^{l,m,n}_{0,1,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{0,1,1}-c^{l,m,n-1}_{0,1,1}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{0,2,1}=\mathrm{minmod}
 *   &\left(c_{0,2,1}^{l,m,n},
 *          \alpha_2\left(c^{l,m+1,n}_{0,1,1}-c^{l,m,n}_{0,1,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{0,1,1}-c^{l,m-1,n}_{0,1,1}\right),
 *   \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m,n+1}_{0,2,0}-c^{l,m,n}_{0,2,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,2,0}-c^{l,m,n-1}_{0,2,0}\right)
 *     \right),
 * \f}
 * \f{align*}{
 * \tilde{c}^{l,m,n}_{2,0,0}=\mathrm{minmod}
 *   &\left(c_{2,0,0}^{l,m,n},
 *          \alpha_2\left(c^{l+1,m,n}_{1,0,0}-c^{l,m,n}_{1,0,0}\right),
 *          \alpha_2\left(c^{l,m,n}_{1,0,0}-c^{l-1,m,n}_{1,0,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{0,2,0}=\mathrm{minmod}
 *   &\left(c_{0,2,0}^{l,m,n},
 *          \alpha_2\left(c^{l,m+1,n}_{0,1,0}-c^{l,m,n}_{0,1,0}\right),
 *          \alpha_2\left(c^{l,m,n}_{0,1,0}-c^{l,m-1,n}_{0,1,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{0,0,2}=\mathrm{minmod}
 *   &\left(c_{0,0,2}^{l,m,n},
 *          \alpha_2\left(c^{l,m,n+1}_{0,0,1}-c^{l,m,n}_{0,0,1}\right),
 *          \alpha_2\left(c^{l,m,n}_{0,0,1}-c^{l,m,n-1}_{0,0,1}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{1,1,1}=\mathrm{minmod}
 *   &\left(c_{1,1,1}^{l,m,n},
 *          \alpha_1\left(c^{l+1,m,n}_{0,1,1}-c^{l,m,n}_{0,1,1}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,1,1}-c^{l-1,m,n}_{0,1,1}\right),
 *     \right.\\
 *   &\;\;\;\;
 *          \alpha_1\left(c^{l,m+1,n}_{1,0,1}-c^{l,m,n}_{1,0,1}\right),
 *          \alpha_1\left(c^{l,m,n}_{1,0,1}-c^{l,m-1,n}_{1,0,1}\right),\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m,n+1}_{1,1,0}-c^{l,m,n}_{1,1,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{1,1,0}-c^{l,m,n-1}_{1,1,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{1,1,0}=\mathrm{minmod}
 *   &\left(c_{1,1,0}^{l,m,n},
 *          \alpha_1\left(c^{l+1,m,n}_{0,1,0}-c^{l,m,n}_{0,1,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,1,0}-c^{l-1,m,n}_{0,1,0}\right),
 *     \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m+1,n}_{1,0,0}-c^{l,m,n}_{1,0,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{1,0,0}-c^{l,m-1,n}_{1,0,0}\right),
 *     \right),\\
 * \tilde{c}^{l,m,n}_{1,0,1}=\mathrm{minmod}
 *   &\left(c_{1,0,1}^{l,m,n},
 *          \alpha_1\left(c^{l+1,m,n}_{0,0,1}-c^{l,m,n}_{0,0,1}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,0,1}-c^{l-1,m,n}_{0,0,1}\right),
 *     \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m,n+1}_{1,0,0}-c^{l,m,n}_{1,0,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{1,0,0}-c^{l,m,n-1}_{1,0,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{0,1,1}=\mathrm{minmod}
 *   &\left(c_{0,1,1}^{l,m,n},
 *          \alpha_1\left(c^{l,m+1,n}_{0,0,1}-c^{l,m,n}_{0,0,1}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,0,1}-c^{l,m-1,n}_{0,0,1}\right),
 *   \right.\\
 *   &\;\;\;\;\left.
 *          \alpha_1\left(c^{l,m,n+1}_{0,1,0}-c^{l,m,n}_{0,1,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,1,0}-c^{l,m,n-1}_{0,1,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{1,0,0}=\mathrm{minmod}
 *   &\left(c_{1,0,0}^{l,m,n},
 *          \alpha_1\left(c^{l+1,m,n}_{0,0,0}-c^{l,m,n}_{0,0,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,0,0}-c^{l-1,m,n}_{0,0,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{0,1,0}=\mathrm{minmod}
 *   &\left(c_{0,1,0}^{l,m,n},
 *          \alpha_1\left(c^{l,m+1,n}_{0,0,0}-c^{l,m,n}_{0,0,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,0,0}-c^{l,m-1,n}_{0,0,0}\right)
 *     \right),\\
 * \tilde{c}^{l,m,n}_{0,0,1}=\mathrm{minmod}
 *   &\left(c_{0,0,1}^{l,m,n},
 *          \alpha_1\left(c^{l,m,n+1}_{0,0,0}-c^{l,m,n}_{0,0,0}\right),
 *          \alpha_1\left(c^{l,m,n}_{0,0,0}-c^{l,m,n-1}_{0,0,0}\right)
 *     \right),
 * \f}
 *
 *
 * The algorithm to perform the limiting is as follows:
 *
 * - limit \f$c_{2,2,2}\f$ (i.e. \f$c_{2,2,2}\leftarrow\tilde{c}_{2,2,2}\f$), if
 *   not changed, stop
 * - limit \f$c_{2,2,1}\f$, \f$c_{2,1,2}\f$, and \f$c_{1,2,2}\f$, if all not
 * changed, stop
 * - limit \f$c_{2,2,0}\f$, \f$c_{2,0,2}\f$, and \f$c_{0,2,2}\f$, if all not
 * changed, stop
 * - limit \f$c_{2,1,1}\f$, \f$c_{1,2,1}\f$, and \f$c_{1,1,2}\f$, if all not
 * changed, stop
 * - limit \f$c_{2,1,0}\f$, \f$c_{2,0,1}\f$, \f$c_{1,2,0}\f$, \f$c_{1,0,2}\f$,
 * \f$c_{0,1,2}\f$, and \f$c_{0,2,1}\f$, if all not changed, stop
 * - limit \f$c_{2,0,0}\f$, \f$c_{0,2,0}\f$, and \f$c_{0,0,2}\f$, if all not
 * changed, stop
 * - limit \f$c_{1,1,1}\f$, if not changed, stop
 * - limit \f$c_{1,1,0}\f$, \f$c_{1,0,1}\f$, and \f$c_{0,1,1}\f$, if all not
 * changed, stop
 * - limit \f$c_{1,0,0}\f$, \f$c_{0,1,0}\f$, and \f$c_{0,0,1}\f$, if all not
 * changed, stop
 *
 * The 1d and 2d implementations are straightforward restrictions of the
 * described algorithm.
 *
 * #### Limitations:
 *
 * - We currently recompute the spectral coefficients for local use, after
 *   having computed these same coefficients for sending to the neighbors. We
 *   should be able to avoid this either by storing the coefficients in the
 *   DataBox or by allowing the limiters' `packaged_data` function to return an
 *   object to be passed as an additional argument to the `operator()` (still
 *   would need to be stored in the DataBox).
 * - We cannot handle the case where neighbors have more/fewer
 *   coefficients. In the case that the neighbor has more coefficients we could
 *   just ignore the higher coefficients. In the case that the neighbor has
 *   fewer coefficients we have a few choices.
 * - Having a different number of collocation points in different directions is
 *   not supported. However, it is straightforward to handle this case. The
 *   outermost loop should be in the direction with the most collocation points,
 *   while the inner most loop should be over the direction with the fewest
 *   collocation points. The highest to lowest coefficients can then be limited
 *   appropriately again.
 * - h-refinement is not supported, but there is one reasonably straightforward
 *   implementation that may work. In this case we would ignore refinement that
 *   is not in the direction of the neighbor, treating the element as
 *   simply having multiple neighbors in that direction. The only change would
 *   be accounting for different refinement in the direction of the neighor,
 *   which should be easy to add since the differences in coefficients in
 *   Eq.\f$\ref{eq:krivodonova 3d minmod}\f$ will just be multiplied by
 *   non-unity factors.
 */
template <size_t VolumeDim, typename... Tags>
class Krivodonova<VolumeDim, tmpl::list<Tags...>> {
 public:
  /*!
   * \brief The \f$\alpha_i\f$ values in the Krivodonova algorithm.
   */
  struct Alphas {
    using type = std::array<
        double, Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>;
    static constexpr OptionString help = {
        "The alpha parameters of the Krivodonova limiter"};
    static type default_value() noexcept {
      return make_array<
          Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>(1.0);
    }
  };
  /*!
   * \brief Turn the limiter off
   *
   * This option exists to temporarily disable the limiter for debugging
   * purposes. For problems where limiting is not needed, the preferred
   * approach is to not compile the limiter into the executable.
   */
  struct DisableForDebugging {
    using type = bool;
    static type default_value() noexcept { return false; }
    static constexpr OptionString help = {"Disable the limiter"};
  };

  using options = tmpl::list<Alphas, DisableForDebugging>;
  static constexpr OptionString help = {
      "The hierarchical slope limiter of Krivodonova.\n\n"
      "This slope limiter works by limiting the highest modal "
      "coefficients/derivatives using an aggressive minmod approach, "
      "decreasing in modal coefficient order until no more limiting is "
      "necessary."};

  explicit Krivodonova(
      std::array<double,
                 Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>
          alphas,
      bool disable_for_debugging = false, const OptionContext& context = {});

  Krivodonova() = default;
  Krivodonova(const Krivodonova&) = delete;
  Krivodonova& operator=(const Krivodonova&) = delete;
  Krivodonova(Krivodonova&&) = default;
  Krivodonova& operator=(Krivodonova&&) = default;
  ~Krivodonova() = default;

  // NOLINTNEXTLINE(google-runtime-reference)
  void pup(PUP::er& p) noexcept;

  bool operator==(const Krivodonova& rhs) const noexcept;

  struct PackagedData {
    Variables<tmpl::list<::Tags::Modal<Tags>...>> modal_volume_data;
    Mesh<VolumeDim> mesh;

    // clang-tidy: google-runtime-references
    void pup(PUP::er& p) noexcept {  // NOLINT
      p | modal_volume_data;
      p | mesh;
    }
  };

  using package_argument_tags = tmpl::list<Tags..., ::Tags::Mesh<VolumeDim>>;

  /// \brief Package data for sending to neighbor elements.
  void package_data(gsl::not_null<PackagedData*> packaged_data,
                    const db::item_type<Tags>&... tensors,
                    const Mesh<VolumeDim>& mesh,
                    const OrientationMap<VolumeDim>& orientation_map) const
      noexcept;

  using limit_tags = tmpl::list<Tags...>;
  using limit_argument_tags =
      tmpl::list<::Tags::Element<VolumeDim>, ::Tags::Mesh<VolumeDim>>;

  bool operator()(
      const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
      const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
      const std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
          neighbor_data) const noexcept;

 private:
  template <typename Tag>
  char limit_one_tensor(
      gsl::not_null<Variables<tmpl::list<::Tags::Modal<Tags>...>>*> coeffs_self,
      gsl::not_null<bool*> limited_any_component, const Mesh<1>& mesh,
      const std::unordered_map<
          std::pair<Direction<1>, ElementId<1>>, PackagedData,
          boost::hash<std::pair<Direction<1>, ElementId<1>>>>& neighbor_data)
      const noexcept;
  template <typename Tag>
  char limit_one_tensor(
      gsl::not_null<Variables<tmpl::list<::Tags::Modal<Tags>...>>*> coeffs_self,
      gsl::not_null<bool*> limited_any_component, const Mesh<2>& mesh,
      const std::unordered_map<
          std::pair<Direction<2>, ElementId<2>>, PackagedData,
          boost::hash<std::pair<Direction<2>, ElementId<2>>>>& neighbor_data)
      const noexcept;
  template <typename Tag>
  char limit_one_tensor(
      gsl::not_null<Variables<tmpl::list<::Tags::Modal<Tags>...>>*> coeffs_self,
      gsl::not_null<bool*> limited_any_component, const Mesh<3>& mesh,
      const std::unordered_map<
          std::pair<Direction<3>, ElementId<3>>, PackagedData,
          boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data)
      const noexcept;

  template <typename Tag, size_t Dim>
  char fill_variables_tag_with_spectral_coeffs(
      gsl::not_null<Variables<tmpl::list<::Tags::Modal<Tags>...>>*>
          modal_coeffs,
      const db::item_type<Tag>& nodal_tensor, const Mesh<Dim>& mesh) const
      noexcept;

  std::array<double,
             Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>
      alphas_ = make_array<
          Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>(1.0);
  bool disable_for_debugging_{false};
};

template <size_t VolumeDim, typename... Tags>
Krivodonova<VolumeDim, tmpl::list<Tags...>>::Krivodonova(
    std::array<double,
               Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>
        alphas,
    bool disable_for_debugging, const OptionContext& context)
    : alphas_(alphas), disable_for_debugging_(disable_for_debugging) {
  // See the main documentation for an explanation of why these bounds are
  // different from those of Krivodonova 2007
  if (alg::any_of(alphas_, [](const double t) noexcept {
        return t > 1.0 or t <= 0.0;
      })) {
    PARSE_ERROR(context,
                "The alphas in the Krivodonova limiter must be in the range "
                "(0,1].");
  }
}

template <size_t VolumeDim, typename... Tags>
void Krivodonova<VolumeDim, tmpl::list<Tags...>>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const db::item_type<Tags>&... tensors, const Mesh<VolumeDim>& mesh,
    const OrientationMap<VolumeDim>& orientation_map) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not initialize packaged_data
    return;
  }

  packaged_data->modal_volume_data.initialize(mesh.number_of_grid_points());
  // perform nodal coefficients to modal coefficients transformation on each
  // tensor component
  expand_pack(fill_variables_tag_with_spectral_coeffs<Tags>(
      &(packaged_data->modal_volume_data), tensors, mesh)...);

  packaged_data->modal_volume_data = orient_variables(
      packaged_data->modal_volume_data, mesh.extents(), orientation_map);

  packaged_data->mesh = orientation_map(mesh);
}

template <size_t VolumeDim, typename... Tags>
bool Krivodonova<VolumeDim, tmpl::list<Tags...>>::operator()(
    const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }
  if (UNLIKELY(mesh != Mesh<VolumeDim>(mesh.extents()[0], mesh.basis()[0],
                                       mesh.quadrature()[0]))) {
    ERROR(
        "The Krivodonova limiter does not yet support non-uniform number of "
        "collocation points, bases, and quadrature in each direction. The "
        "mesh is: "
        << mesh);
  }
  if (UNLIKELY(alg::any_of(element.neighbors(),
                           [](const auto& direction_neighbors) noexcept {
                             return direction_neighbors.second.size() != 1;
                           }))) {
    ERROR("The Krivodonova limiter does not yet support h-refinement");
  }
  alg::for_each(neighbor_data, [&mesh](const auto& id_packaged_data) noexcept {
    if (UNLIKELY(id_packaged_data.second.mesh != mesh)) {
      ERROR(
          "The Krivodonova limiter does not yet support differing meshes "
          "between neighbors. Self mesh is: "
          << mesh << " neighbor mesh is: " << id_packaged_data.second.mesh);
    }
  });

  // Compute local modal coefficients
  Variables<tmpl::list<::Tags::Modal<Tags>...>> coeffs_self(
      mesh.number_of_grid_points(), 0.0);
  expand_pack(fill_variables_tag_with_spectral_coeffs<Tags>(&coeffs_self,
                                                            *tensors, mesh)...);

  // Perform the limiting on the modal coefficients
  bool limited_any_component = false;
  expand_pack(limit_one_tensor<::Tags::Modal<Tags>>(
      make_not_null(&coeffs_self), make_not_null(&limited_any_component), mesh,
      neighbor_data)...);

  // transform back to nodal coefficients
  const auto wrap_copy_nodal_coeffs =
      [&mesh, &coeffs_self ](auto tag, const auto tensor) noexcept {
    auto& coeffs_tensor = get<decltype(tag)>(coeffs_self);
    auto tensor_it = tensor->begin();
    for (auto coeffs_it = coeffs_tensor.begin();
         coeffs_it != coeffs_tensor.end();
         (void)++coeffs_it, (void)++tensor_it) {
      to_nodal_coefficients(make_not_null(&*tensor_it), *coeffs_it, mesh);
    }
    return '0';
  };
  expand_pack(wrap_copy_nodal_coeffs(::Tags::Modal<Tags>{}, tensors)...);

  return limited_any_component;
}

template <size_t VolumeDim, typename... Tags>
template <typename Tag>
char Krivodonova<VolumeDim, tmpl::list<Tags...>>::limit_one_tensor(
    const gsl::not_null<Variables<tmpl::list<::Tags::Modal<Tags>...>>*>
        coeffs_self,
    const gsl::not_null<bool*> limited_any_component, const Mesh<1>& mesh,
    const std::unordered_map<
        std::pair<Direction<1>, ElementId<1>>, PackagedData,
        boost::hash<std::pair<Direction<1>, ElementId<1>>>>& neighbor_data)
    const noexcept {
  using tensor_type = typename Tag::type;
  for (size_t storage_index = 0; storage_index < tensor_type::size();
       ++storage_index) {
    // for each coefficient...
    for (size_t i = mesh.extents()[0] - 1; i > 0; --i) {
      auto& self_coeffs = get<Tag>(*coeffs_self)[storage_index];
      double min_abs_coeff = std::abs(self_coeffs[i]);
      const double sgn_of_coeff = sgn(self_coeffs[i]);
      bool sgns_all_equal = true;
      for (const auto& kv : neighbor_data) {
        const auto& neighbor_coeffs =
            get<Tag>(kv.second.modal_volume_data)[storage_index];
        const double tmp = kv.first.first.sign() * gsl::at(alphas_, i) *
                           (neighbor_coeffs[i - 1] - self_coeffs[i - 1]);

        min_abs_coeff = std::min(min_abs_coeff, std::abs(tmp));
        sgns_all_equal &= sgn(tmp) == sgn_of_coeff;
        if (not sgns_all_equal) {
          self_coeffs[i] = 0.0;
          break;
        }
      }
      if (sgns_all_equal) {
        const double tmp = sgn_of_coeff * min_abs_coeff;
        if (tmp == self_coeffs[i]) {
          break;
        }
        *limited_any_component |= true;
        self_coeffs[i] = tmp;
      }
    }
  }
  return '0';
}

template <size_t VolumeDim, typename... Tags>
template <typename Tag>
char Krivodonova<VolumeDim, tmpl::list<Tags...>>::limit_one_tensor(
    const gsl::not_null<Variables<tmpl::list<::Tags::Modal<Tags>...>>*>
        coeffs_self,
    const gsl::not_null<bool*> limited_any_component, const Mesh<2>& mesh,
    const std::unordered_map<
        std::pair<Direction<2>, ElementId<2>>, PackagedData,
        boost::hash<std::pair<Direction<2>, ElementId<2>>>>& neighbor_data)
    const noexcept {
  using tensor_type = typename Tag::type;
  const auto minmod = [&coeffs_self, &mesh, &neighbor_data, this ](
      const size_t local_i, const size_t local_j,
      const size_t local_tensor_storage_index) noexcept {
    const auto& self_coeffs =
        get<Tag>(*coeffs_self)[local_tensor_storage_index];
    double min_abs_coeff = std::abs(
        self_coeffs[mesh.storage_index(Index<VolumeDim>{local_i, local_j})]);
    const double sgn_of_coeff = sgn(
        self_coeffs[mesh.storage_index(Index<VolumeDim>{local_i, local_j})]);
    for (const auto& kv : neighbor_data) {
      const Direction<2>& dir = kv.first.first;
      const auto& neighbor_coeffs =
          get<Tag>(kv.second.modal_volume_data)[local_tensor_storage_index];
      const size_t index_i =
          dir.axis() == Direction<VolumeDim>::Axis::Xi ? local_i - 1 : local_i;
      const size_t index_j =
          dir.axis() == Direction<VolumeDim>::Axis::Eta ? local_j - 1 : local_j;
      // skip neighbors where we cannot compute a finite difference in that
      // direction because we are already at the lowest coefficient.
      if (index_i == std::numeric_limits<size_t>::max() or
          index_j == std::numeric_limits<size_t>::max()) {
        continue;
      }
      const size_t alpha_index =
          dir.axis() == Direction<VolumeDim>::Axis::Xi ? local_i : local_j;
      const double tmp =
          dir.sign() * gsl::at(alphas_, alpha_index) *
          (neighbor_coeffs[mesh.storage_index(
               Index<VolumeDim>{index_i, index_j})] -
           self_coeffs[mesh.storage_index(Index<VolumeDim>{index_i, index_j})]);

      min_abs_coeff = std::min(min_abs_coeff, std::abs(tmp));
      if (sgn(tmp) != sgn_of_coeff) {
        return 0.0;
      }
    }
    return sgn_of_coeff * min_abs_coeff;
  };

  for (size_t tensor_storage_index = 0;
       tensor_storage_index < tensor_type::size(); ++tensor_storage_index) {
    // for each coefficient...
    for (size_t i = mesh.extents()[0] - 1; i > 0; --i) {
      for (size_t j = i; j < mesh.extents()[1]; --j) {
        // Check if we are done limiting, and if so we move on to the next
        // tensor component.
        auto& self_coeffs = get<Tag>(*coeffs_self)[tensor_storage_index];
        // We treat the different cases separately to reduce the number of
        // times we call minmod, not because it is required for correctness.
        if (UNLIKELY(i == j)) {
          const double tmp = minmod(i, j, tensor_storage_index);
          if (tmp == self_coeffs[mesh.storage_index(Index<VolumeDim>{i, j})]) {
            goto next_tensor_index;
          }
          *limited_any_component |= true;
          self_coeffs[mesh.storage_index(Index<VolumeDim>{i, j})] = tmp;
        } else {
          const double tmp_ij = minmod(i, j, tensor_storage_index);
          const double tmp_ji = minmod(j, i, tensor_storage_index);
          if (tmp_ij ==
                  self_coeffs[mesh.storage_index(Index<VolumeDim>{i, j})] and
              tmp_ji ==
                  self_coeffs[mesh.storage_index(Index<VolumeDim>{j, i})]) {
            goto next_tensor_index;
          }
          *limited_any_component |= true;
          self_coeffs[mesh.storage_index(Index<VolumeDim>{i, j})] = tmp_ij;
          self_coeffs[mesh.storage_index(Index<VolumeDim>{j, i})] = tmp_ji;
        }
      }
    }
  next_tensor_index:
    continue;
  }
  return '0';
}

template <size_t VolumeDim, typename... Tags>
template <typename Tag>
char Krivodonova<VolumeDim, tmpl::list<Tags...>>::limit_one_tensor(
    const gsl::not_null<Variables<tmpl::list<::Tags::Modal<Tags>...>>*>
        coeffs_self,
    const gsl::not_null<bool*> limited_any_component, const Mesh<3>& mesh,
    const std::unordered_map<
        std::pair<Direction<3>, ElementId<3>>, PackagedData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data)
    const noexcept {
  using tensor_type = typename Tag::type;
  const auto minmod = [&coeffs_self, &mesh, &neighbor_data, this ](
      const size_t local_i, const size_t local_j, const size_t local_k,
      const size_t local_tensor_storage_index) noexcept {
    const auto& self_coeffs =
        get<Tag>(*coeffs_self)[local_tensor_storage_index];
    double min_abs_coeff = std::abs(self_coeffs[mesh.storage_index(
        Index<VolumeDim>{local_i, local_j, local_k})]);
    const double sgn_of_coeff = sgn(self_coeffs[mesh.storage_index(
        Index<VolumeDim>{local_i, local_j, local_k})]);
    for (const auto& kv : neighbor_data) {
      const Direction<3>& dir = kv.first.first;
      const auto& neighbor_coeffs =
          get<Tag>(kv.second.modal_volume_data)[local_tensor_storage_index];
      const size_t index_i =
          dir.axis() == Direction<VolumeDim>::Axis::Xi ? local_i - 1 : local_i;
      const size_t index_j =
          dir.axis() == Direction<VolumeDim>::Axis::Eta ? local_j - 1 : local_j;
      const size_t index_k = dir.axis() == Direction<VolumeDim>::Axis::Zeta
                                 ? local_k - 1
                                 : local_k;
      // skip neighbors where we cannot compute a finite difference in that
      // direction because we are already at the lowest coefficient.
      if (index_i == std::numeric_limits<size_t>::max() or
          index_j == std::numeric_limits<size_t>::max() or
          index_k == std::numeric_limits<size_t>::max()) {
        continue;
      }
      const size_t alpha_index =
          dir.axis() == Direction<VolumeDim>::Axis::Xi
              ? local_i
              : dir.axis() == Direction<VolumeDim>::Axis::Eta ? local_j
                                                              : local_k;
      const double tmp = dir.sign() * gsl::at(alphas_, alpha_index) *
                         (neighbor_coeffs[mesh.storage_index(
                              Index<VolumeDim>{index_i, index_j, index_k})] -
                          self_coeffs[mesh.storage_index(
                              Index<VolumeDim>{index_i, index_j, index_k})]);

      min_abs_coeff = std::min(min_abs_coeff, std::abs(tmp));
      if (sgn(tmp) != sgn_of_coeff) {
        return 0.0;
      }
    }
    return sgn_of_coeff * min_abs_coeff;
  };

  for (size_t tensor_storage_index = 0;
       tensor_storage_index < tensor_type::size(); ++tensor_storage_index) {
    // for each coefficient...
    for (size_t i = mesh.extents()[0] - 1; i > 0; --i) {
      for (size_t j = i; j < mesh.extents()[1]; --j) {
        for (size_t k = j; k < mesh.extents()[2]; --k) {
          // Check if we are done limiting, and if so we move on to the next
          // tensor component.
          auto& self_coeffs = get<Tag>(*coeffs_self)[tensor_storage_index];
          // We treat the different cases separately to reduce the number of
          // times we call minmod, not because it is required for correctness.
          //
          // Note that the case `i == k and i != j` cannot be encountered since
          // the loop bounds are `i >= j >= k`.
          if (UNLIKELY(i == j and i == k)) {
            const double tmp = minmod(i, j, k, tensor_storage_index);
            if (tmp ==
                self_coeffs[mesh.storage_index(Index<VolumeDim>{i, j, k})]) {
              goto next_tensor_index;
            }
            *limited_any_component |= true;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{i, j, k})] = tmp;
          } else if (i == j and i != k) {
            const double tmp_ijk = minmod(i, j, k, tensor_storage_index);
            const double tmp_ikj = minmod(i, k, j, tensor_storage_index);
            const double tmp_kij = minmod(k, i, j, tensor_storage_index);
            if (tmp_ijk == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{i, j, k})] and
                tmp_ikj == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{i, k, j})] and
                tmp_kij == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{k, i, j})]) {
              goto next_tensor_index;
            }
            *limited_any_component |= true;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{i, j, k})] =
                tmp_ijk;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{i, k, j})] =
                tmp_ikj;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{k, i, j})] =
                tmp_kij;
          } else if (i != j and j == k) {
            const double tmp_ijk = minmod(i, j, k, tensor_storage_index);
            const double tmp_kij = minmod(k, i, j, tensor_storage_index);
            const double tmp_kji = minmod(k, j, i, tensor_storage_index);
            if (tmp_ijk == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{i, j, k})] and
                tmp_kij == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{k, i, j})] and
                tmp_kji == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{k, j, i})]) {
              goto next_tensor_index;
            }
            *limited_any_component |= true;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{i, j, k})] =
                tmp_ijk;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{k, i, j})] =
                tmp_kij;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{k, j, i})] =
                tmp_kji;
          } else {
            const double tmp_ijk = minmod(i, j, k, tensor_storage_index);
            const double tmp_jik = minmod(j, i, k, tensor_storage_index);
            const double tmp_ikj = minmod(i, k, j, tensor_storage_index);
            const double tmp_jki = minmod(j, k, i, tensor_storage_index);
            const double tmp_kij = minmod(k, i, j, tensor_storage_index);
            const double tmp_kji = minmod(k, j, i, tensor_storage_index);
            if (tmp_ijk == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{i, j, k})] and
                tmp_jik == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{j, i, k})] and
                tmp_ikj == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{i, k, j})] and
                tmp_jki == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{j, k, i})] and
                tmp_kij == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{k, i, j})] and
                tmp_kji == self_coeffs[mesh.storage_index(
                               Index<VolumeDim>{k, j, i})]) {
              goto next_tensor_index;
            }
            *limited_any_component |= true;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{i, j, k})] =
                tmp_ijk;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{j, i, k})] =
                tmp_jik;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{i, k, j})] =
                tmp_ikj;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{j, k, i})] =
                tmp_jki;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{k, i, j})] =
                tmp_kij;
            self_coeffs[mesh.storage_index(Index<VolumeDim>{k, j, i})] =
                tmp_kji;
          }
        }
      }
    }
  next_tensor_index:
    continue;
  }
  return '0';
}

template <size_t VolumeDim, typename... Tags>
template <typename Tag, size_t Dim>
char Krivodonova<VolumeDim, tmpl::list<Tags...>>::
    fill_variables_tag_with_spectral_coeffs(
        const gsl::not_null<Variables<tmpl::list<::Tags::Modal<Tags>...>>*>
            modal_coeffs,
        const db::item_type<Tag>& nodal_tensor, const Mesh<Dim>& mesh) const
    noexcept {
  auto& coeffs_tensor = get<::Tags::Modal<Tag>>(*modal_coeffs);
  auto tensor_it = nodal_tensor.begin();
  for (auto coeffs_it = coeffs_tensor.begin(); coeffs_it != coeffs_tensor.end();
       (void)++coeffs_it, (void)++tensor_it) {
    to_modal_coefficients(make_not_null(&*coeffs_it), *tensor_it, mesh);
  }
  return '0';
}

template <size_t VolumeDim, typename... Tags>
void Krivodonova<VolumeDim, tmpl::list<Tags...>>::pup(PUP::er& p) noexcept {
  p | alphas_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim, typename... Tags>
bool Krivodonova<VolumeDim, tmpl::list<Tags...>>::operator==(
    const Krivodonova<VolumeDim, tmpl::list<Tags...>>& rhs) const noexcept {
  return alphas_ == rhs.alphas_ and
         disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim, typename... Tags>
bool operator!=(
    const Krivodonova<VolumeDim, tmpl::list<Tags...>>& lhs,
    const Krivodonova<VolumeDim, tmpl::list<Tags...>>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace SlopeLimiters
