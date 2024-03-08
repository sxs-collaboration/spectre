// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

/*!
 * \brief Computes `Tags::KleinGordonSource<Tag>` for the tags evolved by
 * Klein-Gordon Cce.
 *
 * \details In scalar-tensor theory, the Cce hypersurface equations get
 * additional source terms contributed by the stress-energy tensor of the scalar
 * field. The tag `Tags::KleinGordonSource<Tag>` stores the corresponding volume
 * data.
 */
template <typename Tag>
struct ComputeKleinGordonSource;

/*!
 * \brief Computes the Klein-Gordon source of the Bondi \f$\beta\f$
 *
 * \details The source reads:
 *
 * \f{align*}{
 * 2 \pi (1-y) (\partial_y\psi)^2,
 * \f}
 * where \f$\psi\f$ is the Klein-Gordon (scalar) field.
 */
template <>
struct ComputeKleinGordonSource<Tags::BondiBeta> {
  using return_tags = tmpl::list<Tags::KleinGordonSource<Tags::BondiBeta>>;
  using argument_tags =
      tmpl::list<Tags::Dy<Tags::KleinGordonPsi>, Tags::OneMinusY>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> kg_source_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_kg_psi,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y);
};

/*!
 * \brief Computes the Klein-Gordon source of the Bondi \f$Q\f$
 *
 * \details Following the nomenclature of \cite Moxon2020gha and their Eq. (49),
 * the scalar field contributes only to the regular part of the source term
 * \f$S_2^R\f$. The expression reads:
 *
 * \f{align*}{
 * 16 \pi \eth\psi \partial_y\psi,
 * \f}
 * where \f$\psi\f$ is the Klein-Gordon (scalar) field.
 */
template <>
struct ComputeKleinGordonSource<Tags::BondiQ> {
  using return_tags = tmpl::list<Tags::KleinGordonSource<Tags::BondiQ>>;
  using argument_tags =
      tmpl::list<Tags::Dy<Tags::KleinGordonPsi>,
                 Spectral::Swsh::Tags::Derivative<Tags::KleinGordonPsi,
                                                  Spectral::Swsh::Tags::Eth>>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> kg_source_q,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_kg_psi,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_kg_psi);
};

/*!
 * \brief Computes the Klein-Gordon source of the Bondi \f$U\f$
 *
 * \details The source vanishes.
 */
template <>
struct ComputeKleinGordonSource<Tags::BondiU> {
  using return_tags = tmpl::list<Tags::KleinGordonSource<Tags::BondiU>>;
  using argument_tags = tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> kg_source_u,
      size_t l_max, size_t number_of_radial_points);
};

/*!
 * \brief Computes the Klein-Gordon source of the Bondi \f$W\f$
 *
 * \details Following the nomenclature of \cite Moxon2020gha and their Eq. (49),
 * the scalar field contributes only to the regular part of the source term
 * \f$S_2^R\f$. The expression reads:
 *
 * \f{align*}{
 * \frac{\pi e^{2\beta}}{R} \left[J(\bar{\eth}\psi)^2 + \bar{J}(\eth
 * \psi)^2-2K \eth\psi\bar{\eth}\psi\right],
 * \f}
 * where \f$\psi\f$ is the Klein-Gordon (scalar) field.
 */
template <>
struct ComputeKleinGordonSource<Tags::BondiW> {
  using return_tags = tmpl::list<Tags::KleinGordonSource<Tags::BondiW>>;
  using argument_tags =
      tmpl::list<Tags::Exp2Beta, Tags::BondiR, Tags::BondiK, Tags::BondiJ,
                 Spectral::Swsh::Tags::Derivative<Tags::KleinGordonPsi,
                                                  Spectral::Swsh::Tags::Eth>>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> kg_source_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_k,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_kg_psi);
};

/*!
 * \brief Computes the Klein-Gordon source of the Bondi \f$H\f$
 *
 * \details Following the nomenclature of \cite Moxon2020gha and their Eq. (50),
 * the scalar field contributes only to the regular part of the source term
 * \f$S_3^R\f$. The expression reads:
 *
 * \f{align*}{
 * 2 \pi \frac{e^{2\beta}}{R}(\eth\psi)^2,
 * \f}
 * where \f$\psi\f$ is the Klein-Gordon (scalar) field.
 */
template <>
struct ComputeKleinGordonSource<Tags::BondiH> {
  using return_tags = tmpl::list<Tags::KleinGordonSource<Tags::BondiH>>;
  using argument_tags =
      tmpl::list<Tags::Exp2Beta, Tags::BondiR,
                 Spectral::Swsh::Tags::Derivative<Tags::KleinGordonPsi,
                                                  Spectral::Swsh::Tags::Eth>>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> kg_source_h,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_kg_psi);
};
}  // namespace Cce
