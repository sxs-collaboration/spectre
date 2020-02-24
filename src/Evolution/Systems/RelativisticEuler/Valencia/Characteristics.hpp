// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace RelativisticEuler {
namespace Valencia {

// @{
/*!
 * \brief Compute the characteristic speeds for the Valencia formulation
 * of the relativistic Euler system.
 *
 * The principal symbol of the system is diagonalized so that the elements of
 * the diagonal matrix are the \f$(\text{Dim} + 2)\f$ characteristic speeds
 *
 * \f{align*}
 * \lambda_1 &= \alpha \Lambda^- - \beta_n,\\
 * \lambda_{i + 1} &= \alpha v_n - \beta_n,\quad i = 1,...,\text{Dim}\\
 * \lambda_{\text{Dim} + 2} &= \alpha \Lambda^+ - \beta_n,
 * \f}
 *
 * where \f$\alpha\f$ is the lapse, \f$\beta_n = n_i \beta^i\f$ and
 * \f$v_n = n_i v^i\f$ are the projections of the shift \f$\beta^i\f$ and the
 * spatial velocity \f$v^i\f$ onto the normal one-form \f$n_i\f$, respectively,
 * and
 *
 * \f{align*}
 * \Lambda^{\pm} &= \dfrac{1}{1 - v^2 c_s^2}\left[ v_n (1- c_s^2) \pm
 * c_s\sqrt{\left(1 - v^2\right)\left[1 - v^2 c_s^2 - v_n^2(1 - c_s^2)\right]}
 * \right],
 * \f}
 *
 * where \f$v^2 = \gamma_{ij}v^iv^j\f$ is the magnitude squared of the spatial
 * velocity, and \f$c_s\f$ is the sound speed.
 */
template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept;

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept;
// @}

/*!
 * \brief Right eigenvectors of the Valencia formulation
 *
 * The principal symbol of the Valencia formulation, \f$A\f$, is diagonalizable,
 * allowing the introduction of characteristic variables \f$U_\text{char}\f$.
 * In terms of the conservative variables \f$U\f$, \f$U = RU_\text{char}\f$ and
 * \f$U_\text{char} = LU\f$, where \f$R\f$ is the matrix whose columns are the
 * right eigenvectors of \f$A\f$, and \f$L = R^{-1}\f$ contains the left
 * eigenvectors as rows. Here we take the ordering
 * \f$U = [\tilde D, \tilde \tau, \tilde S_i]^T\f$. Explicitly,
 * \f$R = [R_-\, R_1\, R_2\, R_3\, R_+]\f$, where [Teukolsky, to be published]
 *
 * \f{align*}
 * R_\pm = \left[\begin{array}{c}
 * 1 \\
 * (hW - 1) \pm \dfrac{hW c_s v_n}{d} \\
 * hW\left(v_i \pm \dfrac{c_s}{d}n_i\right)
 * \end{array}\right],\qquad
 * R_{1,2} = \left[\begin{array}{c}
 * Wv_{(1,2)} \\
 * W\left(2hW - 1\right)v_{(1,2)} \\
 * h\left(t_{(1,2)i} + 2W^2v_{(1,2)}v_i \right)
 * \end{array}\right],\qquad
 * R_3 = \left[\begin{array}{c}
 * K \\
 * K(hW - 1) - hWc_s^2 \\
 * hW(K - c_s^2)v_i
 * \end{array}\right].
 * \f}
 *
 * (One disregards \f$R_2\f$ if working in 2-d, and also \f$R_1\f$ if
 * working in 1-d, so that the number of eigenvectors is
 * always \f$\text{Dim} + 2\f$.) In the above expressions, \f$n_i\f$ is the unit
 * spatial normal along which the characteristic decomposition is carried
 * out, and \f$t_{(1,2)}^i\f$ are unit spatial vectors forming an orthonormal
 * basis with the normal. In addition, \f$h\f$ is the
 * specific enthalpy, \f$W\f$ is the Lorentz factor, \f$v^i\f$ is the spatial
 * velocity, \f$v_n = n_i v^i\f$, \f$v_{(1,2)} = v_i t^i_{(1,2)}\f$,
 * \f$c_s\f$ is the sound speed,
 * \f$d = W\sqrt{1 - v^2c_s^2 - v_n^2(1 - c_s^2)}\f$,
 * \f$K = (1/\rho)(\partial p/\partial\epsilon)_\rho\f$, where \f$\rho\f$ is the
 * rest mass density, \f$p\f$ is the pressure, and \f$\epsilon\f$ is the
 * specific internal energy. Finally, and for numerical purposes, the quantity
 * \f$(hW - 1)\f$ is computed independently using the expression
 *
 * \f{align*}
 * hW - 1 = W\left(\epsilon + \frac{p}{\rho} + \frac{Wv^2}{W + 1}\right),
 * \f}
 *
 * which has a well-behaved Newtonian limit.
 *
 * Following the notation in Valencia::characteristic_speeds, \f$R_\pm\f$
 * are the eigenvectors of the nondegenerate eigenvalues
 * \f$(\alpha\Lambda^\pm - \beta_n)\f$, while \f$ R_{1,2,3}\f$ are the
 * eigenvectors of the Dim-fold degenerate eigenvalue
 * \f$(\alpha v_n - \beta_n)\f$.
 */
template <size_t Dim>
Matrix right_eigenvectors(const Scalar<double>& rest_mass_density,
                          const tnsr::I<double, Dim>& spatial_velocity,
                          const Scalar<double>& specific_internal_energy,
                          const Scalar<double>& pressure,
                          const Scalar<double>& specific_enthalpy,
                          const Scalar<double>& kappa_over_density,
                          const Scalar<double>& sound_speed_squared,
                          const Scalar<double>& lorentz_factor,
                          const tnsr::ii<double, Dim>& spatial_metric,
                          const tnsr::II<double, Dim>& inv_spatial_metric,
                          const Scalar<double>& det_spatial_metric,
                          const tnsr::i<double, Dim>& unit_normal) noexcept;

/*!
 * \brief Left eigenvectors of the Valencia formulation
 *
 * The principal symbol of the Valencia formulation, \f$A\f$, is diagonalizable,
 * allowing the introduction of characteristic variables \f$U_\text{char}\f$.
 * In terms of the conservative variables \f$U\f$, \f$U = RU_\text{char}\f$ and
 * \f$U_\text{char} = LU\f$, where \f$R\f$ is the matrix whose columns are the
 * right eigenvectors of \f$A\f$, and \f$L = R^{-1}\f$ contains the left
 * eigenvectors as rows. Here we take the ordering
 * \f$U = [\tilde D, \tilde \tau, \tilde S_i]^T\f$. Explicitly,
 * \f$L = [L_-\, L_1\, L_2\, L_3\, L_+]^T\f$, where [Teukolsky, to be published]
 *
 * \f{align*}
 * L_\pm &= \dfrac{1}{2h W c_s^2(1 - v_n^2)}\left[\begin{array}{ccc}
 * W(1-v_n^2)\left[c_s^2(h + W) - K(h - W)\right] - c_\pm, &
 * b_\pm, &
 * -a v^i + c_s\left(c_s v_n \pm d\right)n^i
 * \end{array}\right],\\
 * L_{1,2} &= \dfrac{1}{h\left(1 - v_n^2\right)}\left[\begin{array}{ccc}
 * -v_{(1,2)}, & -v_{(1,2)}, & v_{(1,2)}v_n n^i + (1 - v_n^2)t_{(1,2)}^i
 * \end{array}\right],\\
 * L_3 &= \dfrac{1}{hc_s^2}\left[\begin{array}{ccc}
 * h - W, & -W, & Wv^i
 * \end{array}\right],
 * \f}
 *
 * with the definitions
 *
 * \f{align*}
 * d &= W\sqrt{1 - v^2c_s^2 - v_n^2(1 - c_s^2)}, \\
 * a &= W^2\left(1- v_n^2\right)\left(K + c_s^2\right), \\
 * c_\pm &= c_s\left(c_s \pm v_nd\right), \\
 * b_\pm &= a - c_\pm.
 * \f}
 *
 * The rest of the variables are the same as those in
 * Valencia::right_eigenvectors. For numerical purposes, here
 * the quantity \f$(h - W)\f$ is computed independently using the expression
 *
 * \f{align*}
 * h - W =\epsilon + \frac{p}{\rho} - \frac{W^2v^2}{W + 1},
 * \f}
 *
 * which has a well-behaved Newtonian limit.
 *
 * Following the notation in Valencia::characteristic_speeds, \f$L_\pm\f$
 * are the eigenvectors of the nondegenerate eigenvalues
 * \f$(\alpha\Lambda^\pm - \beta_n)\f$, while \f$ L_{1,2,3}\f$ are the
 * eigenvectors of the Dim-fold degenerate eigenvalue
 * \f$(\alpha v_n - \beta_n)\f$.
 */
template <size_t Dim>
Matrix left_eigenvectors(const Scalar<double>& rest_mass_density,
                         const tnsr::I<double, Dim>& spatial_velocity,
                         const Scalar<double>& specific_internal_energy,
                         const Scalar<double>& pressure,
                         const Scalar<double>& specific_enthalpy,
                         const Scalar<double>& kappa_over_density,
                         const Scalar<double>& sound_speed_squared,
                         const Scalar<double>& lorentz_factor,
                         const tnsr::ii<double, Dim>& spatial_metric,
                         const tnsr::II<double, Dim>& inv_spatial_metric,
                         const Scalar<double>& det_spatial_metric,
                         const tnsr::i<double, Dim>& unit_normal) noexcept;

namespace Tags {
template <size_t Dim>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<Dim>;
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<>, gr::Tags::Shift<Dim>,
                 gr::Tags::SpatialMetric<Dim>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SoundSpeedSquared<DataVector>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  using volume_tags = tmpl::list<>;

  using return_type = std::array<DataVector, Dim + 2>;

  static constexpr void function(
      const gsl::not_null<return_type*> result, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift,
      const tnsr::ii<DataVector, Dim>& spatial_metric,
      const tnsr::I<DataVector, Dim>& spatial_velocity,
      const Scalar<DataVector>& sound_speed_squared,
      const tnsr::i<DataVector, Dim>& unit_normal) noexcept {
    characteristic_speeds(
        result, lapse, shift, spatial_velocity,
        dot_product(spatial_velocity, spatial_velocity, spatial_metric),
        sound_speed_squared, unit_normal);
  }
};

}  // namespace Tags

struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<>;
  static double apply() noexcept { return 1.0; }
};

}  // namespace Valencia
}  // namespace RelativisticEuler
