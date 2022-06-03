// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t>
class Direction;
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {

template <size_t NonlinearWeightExponent>
struct Wcns5zReconstructor {
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const q, const int stride, const double epsilon) {
    ASSERT(epsilon > 0.0,
           "epsilon must be greater than zero but is " << epsilon);

    using std::abs;

    const std::array beta{
        1.0833333333333333 * square(q[-2 * stride] - 2.0 * q[-stride] + q[0]) +
            0.25 * square(q[-2 * stride] - 4.0 * q[-stride] + 3.0 * q[0]),
        1.0833333333333333 * square(q[-stride] - 2.0 * q[0] + q[stride]) +
            0.25 * square(q[stride] - q[-stride]),
        1.0833333333333333 * square(q[2 * stride] - 2.0 * q[stride] + q[0]) +
            0.25 * square(q[2 * stride] - 4.0 * q[stride] + 3.0 * q[0])};

    const double tau5{abs(beta[2] - beta[0])};

    const std::array epsilon_k{
        epsilon * (1.0 + abs(q[0]) + abs(q[-stride]) + abs(q[-2 * stride])),
        epsilon * (1.0 + abs(q[0]) + abs(q[-stride]) + abs(q[stride])),
        epsilon * (1.0 + abs(q[0]) + abs(q[stride]) + abs(q[2 * stride]))};

    const std::array nw_buffer{
        1.0 + pow<NonlinearWeightExponent>(tau5 / (beta[0] + epsilon_k[0])),
        1.0 + pow<NonlinearWeightExponent>(tau5 / (beta[1] + epsilon_k[1])),
        1.0 + pow<NonlinearWeightExponent>(tau5 / (beta[2] + epsilon_k[2]))};

    // nonlinear weights for upper and lower reconstructions. The factor 1/16
    // for `alpha`s is omitted here since it is eventually canceled out by
    // denominator when evaluating modified nonlinear weight `omega`s (see the
    // documentation of the `wcns5z()` function below).
    const std::array alpha_upper{nw_buffer[0], 10.0 * nw_buffer[1],
                                 5.0 * nw_buffer[2]};
    const std::array alpha_lower{nw_buffer[2], 10.0 * nw_buffer[1],
                                 5.0 * nw_buffer[0]};
    const double alpha_norm_upper =
        alpha_upper[0] + alpha_upper[1] + alpha_upper[2];
    const double alpha_norm_lower =
        alpha_lower[0] + alpha_lower[1] + alpha_lower[2];

    // reconstruction stencils
    const std::array recons_stencils_upper{
        0.375 * q[-2 * stride] - 1.25 * q[-stride] + 1.875 * q[0],
        -0.125 * q[-stride] + 0.75 * q[0] + 0.375 * q[stride],
        0.375 * q[0] + 0.75 * q[stride] - 0.125 * q[2 * stride]};
    const std::array recons_stencils_lower{
        0.375 * q[2 * stride] - 1.25 * q[stride] + 1.875 * q[0],
        -0.125 * q[stride] + 0.75 * q[0] + 0.375 * q[-stride],
        0.375 * q[0] + 0.75 * q[-stride] - 0.125 * q[-2 * stride]};

    // reconstructed solutions
    return {{(alpha_lower[0] * recons_stencils_lower[0] +
              alpha_lower[1] * recons_stencils_lower[1] +
              alpha_lower[2] * recons_stencils_lower[2]) /
                 alpha_norm_lower,
             (alpha_upper[0] * recons_stencils_upper[0] +
              alpha_upper[1] * recons_stencils_upper[1] +
              alpha_upper[2] * recons_stencils_upper[2]) /
                 alpha_norm_upper}};
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() { return 5; }
};

}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs fifth order weighted compact nonlinear scheme reconstruction
 * based on the WENO-Z method.
 *
 * Using the WENO oscillation indicators given by \cite Jiang1996
 *
 * \f{align}
 *  \beta_0 & = \frac{13}{12} (q_{i-2} - 2q_{i-1} + q_{i})^2
 *              + \frac{1}{4} (q_{i-2} - 4q_{i-1} + 3q_{i})^2  \\
 *  \beta_1 & = \frac{13}{12} (q_{i-1} - 2q_{i} + q_{i+1})^2
 *              + \frac{1}{4} (q_{i+1} - q_{i-1})^2            \\
 *  \beta_2 & = \frac{13}{12} (q_{i} - 2q_{i+1} + q_{i+2})^2
 *              + \frac{1}{4} (3q_{i} - 4q_{i+1} + q_{i+2})^2 ,
 * \f}
 *
 * compute the modified nonlinear weights (cf. WENO-Z scheme: see Eq. 42 of
 * \cite Borges20083191)
 *
 * \f{align}
 *  \omega_k^z = \frac{\alpha_k^z}{\sum_{l=0}^z \alpha_l^z}, \quad
 *  \alpha_k^z = c_k \left(1 + \left(
 *                       \frac{\tau_5}{\beta_k + \epsilon_k} \right)^q \right),
 *  \quad k = 0, 1, 2
 * \f}
 *
 * where \f$\epsilon_k\f$ is a factor used to avoid division by zero and is set
 * to
 *
 * \f{align}
 *  \epsilon_k = \varepsilon (1 + |q_{i+k}| + |q_{i+k-1}| + |q_{i+k-2}|),
 *  \quad k = 0, 1, 2,
 * \f}
 *
 * linear weights \f$c_k\f$ are adopted from \cite Nonomura2013 (see Table 17
 * of it)
 *
 * \f{align}
 *  c_0 = 1/16, \quad c_1 = 10/16, \quad c_2 = 5/16,
 * \f}
 *
 * and \f$\tau_5 \equiv |\beta_2-\beta_0|\f$.
 *
 * Reconstruction stencils are given by Lagrange interpolations (e.g. see Eq.
 * 29d - 29f of \cite Brehm2015)
 * \f{align}
 * q_{i+1/2}^0 &= \frac{3}{8}q_{i-2} -\frac{5}{4}q_{i-1} +\frac{15}{8}q_{i} \\
 * q_{i+1/2}^1 &= -\frac{1}{8}q_{i-1} +\frac{3}{4}q_{i} +\frac{3}{8}q_{i+1} \\
 * q_{i+1/2}^2 &= \frac{3}{8}q_{i} +\frac{3}{4}q_{i+1} -\frac{1}{8}q_{i+2}
 * \f}
 *
 * and the final reconstructed solution is given by
 * \f{align}
 *   q_{i+1/2} = \sum_{k=0}^2 \omega_k q_{i+1/2}^k .
 * \f}
 *
 * The nonlinear weights exponent \f$q (=1 \text{ or } 2)\f$ and the small
 * factor \f$\varepsilon\f$ can be chosen via an input file.
 *
 */
template <size_t NonlinearWeightExponent, size_t Dim>
void wcns5z(const gsl::not_null<std::array<gsl::span<double>, Dim>*>
                reconstructed_upper_side_of_face_vars,
            const gsl::not_null<std::array<gsl::span<double>, Dim>*>
                reconstructed_lower_side_of_face_vars,
            const gsl::span<const double>& volume_vars,
            const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
            const Index<Dim>& volume_extents, const size_t number_of_variables,
            const double epsilon) {
  detail::reconstruct<detail::Wcns5zReconstructor<NonlinearWeightExponent>>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables, epsilon);
}

/*!
 * \brief Returns function pointers to the `wcns5z` function, lower neighbor
 * reconstruction, and upper neighbor reconstruction.
 *
 * This is useful for controlling template parameters like the
 * `NonlinearWeightExponent` from an input file by setting a function pointer.
 * Note that the reason the reconstruction functions instead of say the
 * `pointwise` member function is returned is to avoid function pointers inside
 * tight loops.
 */
template <size_t Dim>
auto wcns5z_function_pointers(size_t nonlinear_weight_exponent) -> std::tuple<
    void (*)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
             gsl::not_null<std::array<gsl::span<double>, Dim>*>,
             const gsl::span<const double>&,
             const DirectionMap<Dim, gsl::span<const double>>&,
             const Index<Dim>&, size_t, double),
    void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
             const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
             const double&),
    void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
             const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
             const double&)>;

}  // namespace fd::reconstruction
