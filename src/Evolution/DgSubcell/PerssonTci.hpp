// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
namespace detail {
template <size_t Dim>
bool persson_tci_impl(gsl::not_null<DataVector*> filtered_component,
                      const DataVector& component, const Mesh<Dim>& dg_mesh,
                      double alpha, double zero_cutoff);
}  // namespace detail

/*!
 * \brief Troubled cell indicator using spectral falloff of \cite Persson2006sub
 *
 * Consider a discontinuity sensing quantity \f$U\f$, which is typically a
 * scalar but could be a tensor of any rank. Let \f$U\f$ have the 1d spectral
 * decomposition (generalization to higher-dimensional tensor product bases is
 * done dimension-by-dimension):
 *
 * \f{align*}{
 *   U(x)=\sum_{i=0}^{N}c_i P_i(x),
 * \f}
 *
 * where \f$P_i(x)\f$ are the basis functions, in our case the Legendre
 * polynomials, and \f$c_i\f$ are the spectral coefficients. We then define a
 * filtered solution \f$\hat{U}\f$ as
 *
 * \f{align*}{
 *   \hat{U}(x)=c_N P_N(x).
 * \f}
 *
 * Note that when an exponential filter is being used to deal with aliasing,
 * lower modes can be included in \f$\hat{U}\f$. The main goal of \f$\hat{U}\f$
 * is to measure how much power is in the highest modes, which are the modes
 * responsible for Gibbs phenomena. We define the discontinuity indicator
 * \f$s^\Omega\f$ as
 *
 * \f{align*}{
 *   s^\Omega=\log_{10}\left(\frac{(\hat{U}, \hat{U})}{(U, U)}\right),
 * \f}
 *
 * where \f$(\cdot,\cdot)\f$ is an inner product, which we take to be the
 * Euclidean \f$L_2\f$ norm (i.e. we do not divide by the number of grid points
 * since that cancels out anyway). A cell is troubled if
 *  \f$s^\Omega > -\alpha \log_{10}(N)\f$. Typically, \f$\alpha=4\f$ is a good
 * choice.
 *
 * The parameter `zero_cutoff` is used to avoid division and logarithms of small
 * numbers, which can be wildly fluctuating because of roundoff errors.
 * We do not check the TCI for tensor components when \f$L_2(\hat{U}) \leq
 * \epsilon L_2(U)\f$, where \f$\epsilon\f$ is the `zero_cutoff`. If all
 * components are skipped the TCI returns `false`, i.e. the cell is not
 * troubled.
 */
template <size_t Dim, typename SymmList, typename IndexList>
bool persson_tci(const Tensor<DataVector, SymmList, IndexList>& tensor,
                 const Mesh<Dim>& dg_mesh, const double alpha,
                 const double zero_cutoff) {
  DataVector filtered_component(dg_mesh.number_of_grid_points());
  for (size_t component_index = 0; component_index < tensor.size();
       ++component_index) {
    if (detail::persson_tci_impl(make_not_null(&filtered_component),
                                 tensor[component_index], dg_mesh, alpha,
                                 zero_cutoff)) {
      return true;
    }
  }
  return false;
}
}  // namespace evolution::dg::subcell
