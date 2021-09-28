// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"               // for get
#include "DataStructures/Tensor/TypeAliases.hpp"          // IWYU pragma: keep
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::MassDensity
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::SoundSpeed
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::SpecificInternalEnergy
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::Velocity
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare hydro::Tags::EquationOfState

/// \cond
class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace NewtonianEuler {

namespace detail {
// Compute the flux Jacobian for the NewtonianEuler system
//
// The flux Jacobian is \f$n_i A^i = n_i \partial F^i / \partial U\f$.
// The input `b_times_theta` is \f$b\theta = \kappa/\rho (v^2 - h) + c_s^2\f$.
//
// This is used for:
// - testing the analytic characteristic transformation
// - as input for the numerical characteristic transformation
template <size_t Dim>
Matrix flux_jacobian(const tnsr::I<double, Dim>& velocity,
                     double kappa_over_density, double b_times_theta,
                     double specific_enthalpy,
                     const tnsr::i<double, Dim>& unit_normal);
}  // namespace detail

/// @{
/*!
 * \brief Compute the characteristic speeds of NewtonianEuler system
 *
 * The principal symbol of the system is diagonalized so that the elements of
 * the diagonal matrix are the characteristic speeds
 *
 * \f{align*}
 * \lambda_1 &= v_n - c_s,\\
 * \lambda_{i + 1} &= v_n,\\
 * \lambda_{\text{Dim} + 2} &= v_n + c_s,
 * \f}
 *
 * where \f$i = 1,...,\text{Dim}\f$,
 * \f$v_n = n_i v^i\f$ is the velocity projected onto the normal,
 * and \f$c_s\f$ is the sound speed.
 */
template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal);

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal);
/// @}

/// @{
/*!
 * \brief Compute the transform matrices between the conserved variables and
 * the characteristic variables of the NewtonianEuler system.
 *
 * Let \f$u\f$ be the conserved (i.e., evolved) variables of the Newtonian Euler
 * system, and \f$w\f$ the characteristic variables of this system with respect
 * to a unit normal one form \f$n_i\f$.  The function `left_eigenvectors`
 * computes the matrix \f$\Omega_{L}\f$ corresponding to the transform
 * \f$w = \Omega_{L} u\f$. The function `right_eigenvectors` computes the matrix
 * \f$\Omega_{R}\f$ corresponding to the inverse transform
 * \f$u = \Omega_{R} w\f$. Here the components of \f$u\f$ are ordered as
 * \f$u = \{\rho, \rho v_x, \rho v_y, \rho v_z, e\}\f$ in 3D, and the components
 * of \f$w\f$ are ordered by their corresponding eigenvalues
 * (i.e., characteristic speeds)
 * \f$\lambda = \{v_n - c_s, v_n, v_n, v_n, v_n + c_s\}\f$. In these
 * expressions, \f$\rho\f$ is the fluid mass density, \f$v_{x,y,z}\f$ are the
 * components of the fluid velocity, \f$e\f$ is the total energy density,
 * \f$v_n\f$ is the component of the velocity along the unit normal \f$n_i\f$,
 * and \f$c_s\f$ is the sound speed.
 *
 * For a short discussion of the characteristic transformation and the matrices
 * \f$\Omega_{L}\f$ and \f$\Omega_{R}\f$, see \cite Kulikovskii2000 Chapter 3.
 *
 * Here we briefly summarize the procedure. With \f$F^x(u)\f$ the Newtonian
 * Euler flux in direction \f$x\f$, then the flux Jacobian along \f$x\f$ is the
 * matrix \f$A_x = \partial F^x_{\beta}(u) / \partial u_{\alpha}\f$. The indices
 * \f$\alpha, \beta\f$ range over the different evolved fields. In higher
 * dimensions, the flux Jacobian along the unit normal \f$n_i\f$ is
 * \f$A = n_x A_x + n_y A_y + n_z A_z\f$.
 * This matrix can be diagonalized as \f$A = \Omega_{R} \Lambda \Omega_{L}\f$.
 * Here \f$\Lambda = \mathrm{diag}(v_n - c_s, v_n, v_n, v_n, v_n + c_s)\f$
 * is a diagonal matrix containing the characteristic speeds; \f$\Omega_{R}\f$
 * is a matrix whose columns are the right eigenvectors of \f$A\f$;
 * \f$\Omega_{L}\f$ is the inverse of \f$R\f$.
 */
template <size_t Dim>
Matrix right_eigenvectors(const tnsr::I<double, Dim>& velocity,
                          const Scalar<double>& sound_speed_squared,
                          const Scalar<double>& specific_enthalpy,
                          const Scalar<double>& kappa_over_density,
                          const tnsr::i<double, Dim>& unit_normal);

template <size_t Dim>
Matrix left_eigenvectors(const tnsr::I<double, Dim>& velocity,
                         const Scalar<double>& sound_speed_squared,
                         const Scalar<double>& specific_enthalpy,
                         const Scalar<double>& kappa_over_density,
                         const tnsr::i<double, Dim>& unit_normal);
/// @}

/*!
 * \brief Compute the transform matrices between the conserved variables and
 * the characteristic variables of the NewtonianEuler system.
 *
 * See `right_eigenvectors` and `left_eigenvectors` for more details.
 *
 * However, note that this function computes the transformation (i.e., the
 * eigenvectors of the flux Jacobian) numerically, instead of using the analytic
 * expressions. This is useful as a proof-of-concept for more complicated
 * systems where the analytic expressions may not be known.
 */
template <size_t Dim>
std::pair<DataVector, std::pair<Matrix, Matrix>> numerical_eigensystem(
    const tnsr::I<double, Dim>& velocity,
    const Scalar<double>& sound_speed_squared,
    const Scalar<double>& specific_enthalpy,
    const Scalar<double>& kappa_over_density,
    const tnsr::i<double, Dim>& unit_normal);

namespace Tags {

template <size_t Dim>
struct CharacteristicSpeedsCompute : CharacteristicSpeeds<Dim>, db::ComputeTag {
  using base = CharacteristicSpeeds<Dim>;
  using argument_tags =
      tmpl::list<Velocity<DataVector, Dim>, SoundSpeed<DataVector>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  using return_type = std::array<DataVector, Dim + 2>;

  static constexpr void function(const gsl::not_null<return_type*> result,
                                 const tnsr::I<DataVector, Dim>& velocity,
                                 const Scalar<DataVector>& sound_speed,
                                 const tnsr::i<DataVector, Dim>& normal) {
    characteristic_speeds<Dim>(result, velocity, sound_speed, normal);
  }
};

struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

template <size_t Dim>
struct ComputeLargestCharacteristicSpeed : LargestCharacteristicSpeed,
                                           db::ComputeTag {
  using argument_tags =
      tmpl::list<Tags::Velocity<DataVector, Dim>, Tags::SoundSpeed<DataVector>>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(const gsl::not_null<double*> speed,
                       const tnsr::I<DataVector, Dim>& velocity,
                       const Scalar<DataVector>& sound_speed) {
    *speed = max(get(magnitude(velocity)) + get(sound_speed));
  }
};
}  // namespace Tags
}  // namespace NewtonianEuler
