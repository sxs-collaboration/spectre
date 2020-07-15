// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"

namespace Cce {

/// The tags that are needed to be interpolated at scri+ for the available
/// observation tags.
using scri_plus_interpolation_set =
    tmpl::list<Tags::News, Tags::ScriPlus<Tags::Strain>,
               Tags::ScriPlus<Tags::Psi3>, Tags::ScriPlus<Tags::Psi2>,
               Tags::ScriPlus<Tags::Psi1>, Tags::ScriPlus<Tags::Psi0>,
               Tags::Du<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>,
               Tags::EthInertialRetardedTime>;

template <typename Tag>
struct CalculateScriPlusValue;

/*!
 * \brief Compute the Bondi news from the evolution quantities.
 *
 * \details In the gauge used for regularity-preserving CCE,
 * the Bondi news takes the convenient form
 *
 * \f{align*}{
 * N = e^{-2 \beta^{(0)}} \left( (\partial_u \bar J)^{(1)}
 * + \bar \eth \bar \eth e^{2 \beta^{(0)}}\right),
 * \f}
 *
 * where \f$(0)\f$ and \f$(1)\f$ in the superscripts denote the zeroth and first
 * order in an expansion in \f$1/r\f$ near \f$\mathcal{I}^+\f$.
 */
template <>
struct CalculateScriPlusValue<Tags::News> {
  using return_tags = tmpl::list<Tags::News>;
  // extra typelist for more convenient testing
  using tensor_argument_tags =
      tmpl::list<Tags::Dy<Tags::Du<Tags::BondiJ>>, Tags::BondiBeta,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::EthEth>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>;
  using argument_tags =
      tmpl::append<tensor_argument_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_du_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_eth_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

/*!
 * \brief Compute the contribution to the leading \f$\Psi_4\f$ that corresponds
 * to a total time derivative.
 *
 * \details The value \f$\Psi_4\f$ scales asymptotically as \f$r^{-1}\f$, and
 * has the form
 *
 * \f{align*}{
 * \Psi_4^{(1)} = \partial_{u_{\text{inertial}}} B,
 * \f}
 *
 * where superscripts denote orders in the expansion in powers of \f$r^{-1}\f$.
 * This mutator computes \f$B\f$:
 *
 * \f{align*}{
 * B = 2 e^{-2 \beta^{(0)}} (\bar \eth \bar U^{(1)} + \partial_u \bar J^{(1)})
 * \f}
 *
 * and the time derivative that appears the original equation obeys,
 *
 * \f[
 * \partial_{u_{\text{inertial}}} = e^{-2 \beta} \partial_u
 * \f]
 */
template <>
struct CalculateScriPlusValue<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>> {
  using return_tags =
      tmpl::list<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>;
  // extra typelist for more convenient testing
  using tensor_argument_tags =
      tmpl::list<Tags::Exp2Beta, Tags::Dy<Tags::BondiU>,
                 Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiU>,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::Dy<Tags::Du<Tags::BondiJ>>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
                 Tags::EthRDividedByR>;
  using argument_tags =
      tmpl::append<tensor_argument_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*>
          integral_of_psi_4,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_bondi_u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_dy_bondi_u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_du_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      size_t l_max, size_t number_of_radial_points) noexcept;
};


/*!
 * \brief Computes the leading part of \f$\Psi_3\f$ near \f$\mathcal I^+\f$.
 *
 * \details The value \f$\Psi_3\f$ scales asymptotically as \f$r^{-2}\f$, and
 * has the form (in the coordinates used for regularity preserving CCE)
 *
 * \f{align*}{
 * \Psi_3^{(2)} = 2 \bar \eth \beta^{(0)}
 * + 4 \bar \eth \beta^{(0)} \eth \bar \eth \beta^{(0)}
 * + \bar \eth  \eth \bar \eth \beta^{(0)}
 * + \frac{e^{-2  \beta^{(0)}}}{2}  \eth \partial_u \bar J^{(1)}
 * - e^{-2  \beta^{(0)}}  \eth  \beta^{(0)}  \partial_u \bar J^{(1)}
 * \f},
 *
 * where \f$J^{(n)}\f$ is the \f$1/r^n\f$ part of \f$J\f$ evaluated at
 * \f$\mathcal I^+\f$, so
 *
 * \f{align*}{
 * J^{(1)} = (-2 R \partial_y J)|_{y = 1},
 * \f}
 *
 * where the expansion is determined by the conversion between Bondi and
 * numerical radii \f$r = 2 R / (1 - y)\f$.
 */
template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi3>> {
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Psi3>>;
  // extra typelist for more convenient testing
  using tensor_argument_tags = tmpl::list<
      Tags::Exp2Beta,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::EthEthbar>,
      Spectral::Swsh::Tags::Derivative<
          Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                           Spectral::Swsh::Tags::EthEthbar>,
          Spectral::Swsh::Tags::Ethbar>,
      Tags::Dy<Tags::Du<Tags::BondiJ>>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::Du<Tags::BondiJ>>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::EthRDividedByR>;
  using argument_tags = tmpl::push_back<tensor_argument_tags, Tags::LMax,
                                        Tags::NumberOfRadialPoints>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -1>>*> psi_3,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& eth_ethbar_beta,
      const Scalar<SpinWeighted<ComplexDataVector, -1>>& ethbar_eth_ethbar_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_du_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& ethbar_dy_du_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

/*!
 * \brief Computes the leading part of \f$\Psi_2\f$ near \f$\mathcal I^+\f$.
 *
 * \details The value \f$\Psi_2\f$ scales asymptotically as \f$r^{-3}\f$, and
 * has the form (in the coordinates used for regularity preserving CCE)
 *
 * \f{align*}{
 * \Psi_2^{(3)} = -\frac{e^{-2  \beta^{(0)}}}{4}
 * \left(e^{2 \beta^{(0)}} \eth \bar Q^{(1)} +  \eth \bar U^{(2)}
 * + \bar \eth  U^{(2)} + J^{(1)} \bar \eth \bar U^{(1)}
 * +  J^{(1)} \bar \partial_u J^{(1)} - 2 W^{(2)}\right)
 * \f},
 *
 * where \f$A^{(n)}\f$ is the \f$1/r^n\f$ part of \f$A\f$ evaluated at
 * \f$\mathcal I^+\f$, so for any quantity \f$A\f$,
 *
 * \f{align*}{
 * \eth  A^{(1)} &= (-2 R \eth  \partial_y  A
 * - 2 \eth R \partial_y  A)|_{y = 1} \notag\\
 * \eth  A^{(2)} &= (2 R^2 \eth \partial_y^2 A
 * + 2 R \eth R \partial^2_y  A)|_{y = 1}, \notag\\
 * A^{(1)} &= (- 2 R \partial_y A)|_{y = 1}, \notag\\
 * A^{(2)} &= (2 R^2 \partial_y^2 A)|_{y = 1},
 * \f}
 *
 * where the expansion is determined by the conversion between Bondi and
 * numerical radii \f$r = 2 R / (1 - y)\f$.
 */
template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi2>> {
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Psi2>>;
  // extra typelist for more convenient testing
  using tensor_argument_tags = tmpl::list<
      Tags::Exp2Beta, Tags::Dy<Tags::BondiQ>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiQ>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Tags::Dy<Tags::BondiU>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiU>,
                                       Spectral::Swsh::Tags::Eth>,
      Tags::Dy<Tags::Dy<Tags::BondiU>>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::Dy<Tags::BondiU>>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Tags::Dy<Tags::Dy<Tags::BondiW>>, Tags::Dy<Tags::BondiJ>,
      Tags::Dy<Tags::Du<Tags::BondiJ>>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::EthRDividedByR>;
  using argument_tags = tmpl::push_back<tensor_argument_tags, Tags::LMax,
                                        Tags::NumberOfRadialPoints>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> psi_2,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_bondi_q,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_dy_bondi_q,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_bondi_u,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_dy_bondi_u,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_dy_bondi_u,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& ethbar_dy_dy_bondi_u,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_dy_bondi_w,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_du_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

/*!
 * \brief Computes the leading part of \f$\Psi_1\f$ near \f$\mathcal I^+\f$.
 *
 * \details The value \f$\Psi_1\f$ scales asymptotically as \f$r^{-4}\f$, and
 * has the form (in the coordinates used for regularity preserving CCE)
 *
 * \f{align*}{
 * \Psi_1^{(4)} = \frac{1}{8} \left(- 12 \eth \beta^{(2)} + J^{(1)} \bar Q^{(1)}
 * + 2 Q^{(2)}\right)
 * \f}
 *
 * where \f$A^{(n)}\f$ is the \f$1/r^n\f$ part of \f$A\f$ evaluated at
 * \f$\mathcal I^+\f$, so for any quantity \f$A\f$,
 *
 * \f{align*}{
 * \eth A^{(2)} &= (2 R^2 \eth \partial_y^2 A
 * + 2 R \eth R \partial^2_y  A)|_{y = 1}, \notag\\
 * A^{(1)} &= (- 2 R \partial_y A)|_{y = 1}, \notag\\
 * A^{(2)} &= (2 R^2 \partial_y^2 A)|_{y = 1},
 * \f}
 *
 * where the expansion is determined by the conversion between Bondi and
 * numerical radii \f$r = 2 R / (1 - y)\f$.
 */
template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi1>> {
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Psi1>>;
  // extra typelist for more convenient testing
  using tensor_argument_tags = tmpl::list<
      Tags::Dy<Tags::Dy<Tags::BondiBeta>>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::Dy<Tags::BondiBeta>>,
                                       Spectral::Swsh::Tags::Eth>,
      Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::BondiQ>,
      Tags::Dy<Tags::Dy<Tags::BondiQ>>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>, Tags::EthRDividedByR>;
  using argument_tags = tmpl::push_back<tensor_argument_tags, Tags::LMax,
                                        Tags::NumberOfRadialPoints>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> psi_1,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& dy_dy_bondi_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_dy_dy_bondi_beta,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_bondi_q,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& dy_dy_bondi_q,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& eth_r_divided_by_r,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

/*!
 * \brief Computes the leading part of \f$\Psi_0\f$ near \f$\mathcal I^+\f$.
 *
 * \details The value \f$\Psi_0\f$ scales asymptotically as \f$r^{-5}\f$, and
 * has the form (in the coordinates used for regularity preserving CCE)
 *
 * \f{align*}{
 * \Psi_0^{(5)} = \frac{3}{2}\left(\frac{1}{4}\bar J^{(1)} J^{(1)} {}^2
 * - J^{(3)}\right)
 * \f}
 *
 * where \f$A^{(n)}\f$ is the \f$1/r^n\f$ part of \f$A\f$ evaluated at
 * \f$\mathcal I^+\f$, so for any quantity \f$A\f$,
 *
 * \f{align*}{
 * A^{(1)} &= (- 2 R \partial_y A)|_{y = 1} \notag\\
 * A^{(3)} &= \left(-\frac{4}{3} R^3 \partial_y^3 A\right)|_{y = 1},
 * \f}
 *
 * where the expansion is determined by the conversion between Bondi and
 * numerical radii \f$r = 2 R / (1 - y)\f$.
 */
template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi0>> {
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Psi0>>;
  // extra typelist for more convenient testing
  using tensor_argument_tags =
      tmpl::list<Tags::Dy<Tags::BondiJ>,
                 Tags::Dy<Tags::Dy<Tags::Dy<Tags::BondiJ>>>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>;
  using argument_tags = tmpl::push_back<tensor_argument_tags, Tags::LMax,
                                        Tags::NumberOfRadialPoints>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi_0,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_dy_dy_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

/*!
 * \brief Computes the leading part of the strain \f$h\f$ near \f$\mathcal
 * I^+\f$.
 *
 * \details The value \f$h\f$ scales asymptotically as \f$r^{-1}\f$, and
 * has the form (in the coordinates used for regularity preserving CCE)
 *
 * \f{align*}{
 * h = \bar J^{(1)} + \bar \eth \bar \eth u^{(0)},
 * \f}
 *
 * where \f$u^{(0)}\f$ is the asymptotically inertial retarded time, and
 * \f$A^{(n)}\f$ is the \f$1/r^n\f$ part of \f$A\f$ evaluated at
 * \f$\mathcal I^+\f$, so for any quantity \f$A\f$,
 *
 * \f{align*}{
 * A^{(1)} = (- 2 R \partial_y A)|_{y = 1},
 * \f}
 *
 * where the expansion is determined by the conversion between Bondi and
 * numerical radii \f$r = 2 R / (1 - y)\f$.
 */
template <>
struct CalculateScriPlusValue<Tags::ScriPlus<Tags::Strain>> {
  using return_tags = tmpl::list<Tags::ScriPlus<Tags::Strain>>;
  // extra typelist for more convenient testing
  using tensor_argument_tags = tmpl::list<
      Tags::Dy<Tags::BondiJ>,
      Spectral::Swsh::Tags::Derivative<Tags::ComplexInertialRetardedTime,
                                       Spectral::Swsh::Tags::EthEth>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>;
  using argument_tags = tmpl::push_back<tensor_argument_tags, Tags::LMax,
                                        Tags::NumberOfRadialPoints>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> strain,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_bondi_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& eth_eth_retarded_time,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

/*!
 * \brief Assign the time derivative of the asymptotically inertial time
 * coordinate.
 *
 * \details The asymptotically inertial time coordinate \f$\mathring u\f$ obeys
 * the differential equation:
 *
 * \f{align*}{
 * \partial_u \mathring u = e^{2 \beta}.
 * \f}
 */
template <>
struct CalculateScriPlusValue<::Tags::dt<Tags::InertialRetardedTime>> {
  using return_tags = tmpl::list<::Tags::dt<Tags::InertialRetardedTime>>;
  using argument_tags = tmpl::list<Tags::Exp2Beta>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> dt_inertial_time,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& exp_2_beta) noexcept;
};

/// Determines the angular derivative of the asymptotic inertial time, useful
/// for asymptotic coordinate transformations.
template <>
struct CalculateScriPlusValue<Tags::EthInertialRetardedTime> {
  using return_tags = tmpl::list<Tags::EthInertialRetardedTime>;
  using argument_tags =
      tmpl::list<Tags::ComplexInertialRetardedTime, Tags::LMax>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          eth_inertial_time,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& inertial_time,
      size_t l_max) noexcept;
};

/// Initialize the \f$\mathcal I^+\f$ value `Tag` for the first hypersurface.
template <typename Tag>
struct InitializeScriPlusValue;

/// Initialize the inertial retarded time to the value provided in the mutator
/// arguments.
template <>
struct InitializeScriPlusValue<Tags::InertialRetardedTime> {
  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<Tags::InertialRetardedTime>;

  static void apply(const gsl::not_null<Scalar<DataVector>*> inertial_time,
                    const double initial_time = 0.0) noexcept {
    // this is arbitrary, and has to do with choosing a BMS frame.
    get(*inertial_time) = initial_time;
  }
};
}  // namespace Cce
