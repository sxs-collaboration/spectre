// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
template <typename T>
class DataVectorImpl;
using DataVector = DataVectorImpl<double>;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace grmhd {
namespace ValenciaDivClean {

/*!
 * \brief Compute the characteristic speeds for the Valencia formulation of
 * GRMHD with divergence cleaning.
 *
 * Obtaining the exact form of the characteristic speeds involves the solution
 * of a nontrivial quartic equation for the fast and slow modes. Here we make
 * use of a common approximation in the literature (e.g. \ref char_ref "[1]")
 * where the resulting characteristic speeds are analogous to those of the
 * Valencia formulation of the 3-D relativistic Euler system
 * (see RelativisticEuler::Valencia::characteristic_speeds),
 *
 * \f{align*}
 * \lambda_2 &= \alpha \Lambda^- - \beta_n,\\
 * \lambda_{3, 4, 5, 6, 7} &= \alpha v_n - \beta_n,\\
 * \lambda_{8} &= \alpha \Lambda^+ - \beta_n,
 * \f}
 *
 * with the substitution
 *
 * \f{align*}
 * c_s^2 \longrightarrow c_s^2 + v_A^2(1 - c_s^2)
 * \f}
 *
 * in the definition of \f$\Lambda^\pm\f$. Here \f$v_A\f$ is the Alfvén
 * speed. In addition, two more speeds corresponding to the divergence cleaning
 * mode and the longitudinal magnetic field are added,
 *
 * \f{align*}
 * \lambda_1 = -\alpha - \beta_n,\\
 * \lambda_9 = \alpha - \beta_n.
 * \f}
 *
 * \note The ordering assumed here is such that, in the Newtonian limit,
 * the exact expressions for \f$\lambda_{2, 8}\f$, \f$\lambda_{3, 7}\f$,
 * and \f$\lambda_{4, 6}\f$ should reduce to the
 * corresponding fast modes, Alfvén modes, and slow modes, respectively.
 * See \ref mhd_ref "[2]" for a detailed description of the hyperbolic
 * characterization of Newtonian MHD.
 *
 * \anchor char_ref [1] C.F Gammie, J.C McKinney, G. Tóth, HARM: A Numerical
 * Scheme for General Relativistic Magnetohydrodynamics, ApJ.
 * [589 (2003) 444](http://iopscience.iop.org/article/10.1086/374594/meta)
 *
 * \anchor mhd_ref [2] A. Dedner et al., Hyperbolic Divergence Cleaning for the
 * MHD Equations, J. Comput. Phys.
 * [175 (2002) 645](https://doi.org/10.1006/jcph.2001.6961)
 */
std::array<DataVector, 9> characteristic_speeds(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& alfven_speed_squared,
    const tnsr::i<DataVector, 3>& normal) noexcept;

}  // namespace ValenciaDivClean
}  // namespace grmhd
