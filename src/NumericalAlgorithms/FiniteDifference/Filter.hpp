// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Structure/DirectionMap.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace fd {
/*!
 * \brief Apply a low-pass filter to the data.
 *
 * The filter fits a Legendre polynomial of degree equal to `fd_order` to each
 * grid point and its neighboring points, then subtracts out the highest mode
 * contribution at the grid point. This is inspired by a Heaviside filter for
 * Legendre polynomial-based spectral methods.
 *
 * The filter at different orders is given by:
 *
 * \f{align*}{
 * F^{(9)} u_i&= -\frac{35 \times 531441}{128} \left(
 *                \frac{1}{6406400} (u_{i-4} + u_{i+4}) -
 *                \frac{1}{800800} (u_{i-3} + u_{i+3}) +
 *                \frac{1}{228800} (u_{i-2} + u_{i+2}) -
 *                \frac{1}{114400} (u_{i-1} + u_{i+1}) +
 *                \frac{1}{91520} u_i\right) \\
 * F^{(7)} u_i&= \frac{5 \times 16807}{16} \left(
 *                \frac{1}{95040} (u_{i-3} + u_{i+3}) -
 *                \frac{1}{15840} (u_{i-2} + u_{i+2}) +
 *                \frac{1}{6336} (u_{i-1} + u_{i+1}) -
 *                \frac{1}{4572} u_i\right) \\
 * F^{(5)} u_i&= -\frac{3 \times 125}{8} \left(
 *                \frac{1}{336} (u_{i-2} + u_{i+2}) -
 *                \frac{1}{84} (u_{i-1} + u_{i+1}) +
 *                \frac{1}{56} u_i\right) \\
 * F^{(3)} u_i&= \frac{1 \times 3}{2} \left(
 *                \frac{1}{4} (u_{i-1} + u_{i+1}) -
 *                \frac{1}{2} u_i\right)
 * \f}
 *
 * \note The \f$F^{(11)}\f$ filter isn't implemented yet.
 *
 * \note The argument \f$\epsilon\f$ controls how much of the mode is filter
 * out. \f$\epsilon=1\f$ means the highest mode is completely filtered while
 * \f$\epsilon=0\f$ means it's not at all filtered.
 */
template <size_t Dim>
void low_pass_filter(
    gsl::not_null<gsl::span<double>*> filtered_data,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, size_t number_of_variables, size_t fd_order,
    double epsilon);

/*!
 * \brief Apply Kreiss-Oliger dissipation \cite Kreiss1973 of order `fd_order`
 * to the variables.
 *
 * Define the operators \f$D_+\f$ and \f$D_-\f$ be defined as:
 *
 * \f{align*}{
 * D_+f_i&=\frac{(f_{i+1}-f_i)}{\Delta x} \\
 * D_-f_i&=\frac{(f_i-f_{i-1})}{\Delta x}
 * \f}
 *
 * where the subscript \f$i\f$ refers to the grid index. The dissipation
 * operators are generally applied dimension-by-dimension, and so we have:
 *
 * \f{align*}{
 * \mathcal{D}^{(2m)}=-\frac{(-1)^m}{2^{2m}}\Delta x^{2m-1}
 * \epsilon(D_{+})^m(D_{-})^m
 * \f}
 *
 * where \f$\epsilon\f$ controls the amount of dissipation and is restricted to
 * \f$0\leq\epsilon\leq1\f$, and \f$m\f$ is the order of the finite difference
 * derivative so as not to spoil the accuracy of the scheme. That is, for
 * second order FD, \f$m=2\f$ and one should use \f$\mathcal{D}^{(4)}\f$.
 * However, this choice requires a larger stencil and whether or not this is
 * necessary also depends on when and how in the algorithm the operators are
 * applied.
 *
 * We arrive at the following operators:
 *
 * \f{align*}{
 *  \mathcal{D}^{(2)}f_i&=-\frac{\epsilon}{\Delta x}
 *   (f_{i+1} - 2f_i+f_{i-1}) \\
 *  \mathcal{D}^{(4)}f_i&=-\frac{\epsilon}{16\Delta x}
 *   (f_{i+2}-4f_{i+1}+6f_i-4f_{i-1}+f_{i-2}) \\
 *  \mathcal{D}^{(6)}f_i&=\frac{\epsilon}{64\Delta x}(f_{i+3}-6f_{i+2}+15f_{i+1}
 *    -20f_i+15f_{i-1}-6f_{i-2}+f_{i-3}) \\
 *  \mathcal{D}^{(8)}f_i&=-\frac{\epsilon}{256\Delta x}
 *    (f_{i+4}-8f_{i+3}+28f_{i+2}-56f_{i+1}+70f_{i}-56f_{i-1}+28f_{i-2}
 *    -8f_{i-3}+f_{i-4}) \\
 *  \mathcal{D}^{(10)}f_i&=\frac{\epsilon}{1024\Delta x}
 *    (f_{i+5}-10f_{i+4}+45f_{i+3}
 *    -120f_{i+2}+210f_{i+1}-252f_{i}+210f_{i-1}-120f_{i-2}+45f_{i-3}
 *    -10f_{i-4}+f_{i-5})
 * \f}
 *
 * \note This function applies \f$\Delta x \mathcal{D}^{(2m)}\f$.
 */
template <size_t Dim>
void kreiss_oliger_filter(
    gsl::not_null<gsl::span<double>*> filtered_data,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, size_t number_of_variables, size_t fd_order,
    double epsilon);
}  // namespace fd
