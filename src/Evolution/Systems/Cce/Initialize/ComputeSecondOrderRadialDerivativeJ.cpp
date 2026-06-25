// Distributed under the MIT License. See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ComputeSecondOrderRadialDerivativeJ.hpp"

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ::CauchySecondOrder_detail {

namespace {
namespace SwshTags = ::Spectral::Swsh::Tags;

// `ApplySwshJacobianInplace` is keyed on a `Derivative` prefix tag that only
// reads off the spin weight of its argument tag. `TempSpinWeightedScalar` lets
// us name a derivative tag of any spin weight, so the same volume machinery
// converts the worldtube angular derivatives of arbitrary spin-weighted
// operands from the numerical (constant y) to the physical (constant r)
// coordinate.
template <int Spin, typename Kind>
using Jacobian = ApplySwshJacobianInplace<
    SwshTags::Derivative<::Tags::TempSpinWeightedScalar<0, Spin>, Kind>>;

// Every angular temporary of `evaluate_worldtube_h_residual` lives in a single
// `Variables` so that the whole computation performs one allocation instead of
// one per temporary. `Temp<Index, Spin>` names the slots; the raw
// `ComplexDataVector` products are stored in spin-0 slots and accessed through
// `.data()`.
template <size_t Index, int Spin>
using Temp = ::Tags::TempSpinWeightedScalar<Index, Spin>;

using BufferTags =
    tmpl::list<Temp<0, 0>, Temp<1, 1>, Temp<2, 0>, Temp<3, 2>, Temp<4, 0>,
               Temp<5, 0>, Temp<6, 0>, Temp<7, 0>, Temp<8, 1>, Temp<9, 1>,
               Temp<10, 1>, Temp<11, 1>, Temp<12, -1>, Temp<13, 3>, Temp<14, 2>,
               Temp<15, 0>, Temp<16, 1>, Temp<17, 0>, Temp<18, 1>, Temp<19, 1>,
               Temp<20, 2>, Temp<21, 0>, Temp<22, 1>, Temp<23, 1>, Temp<24, 1>,
               Temp<25, 1>, Temp<26, 2>, Temp<27, 0>, Temp<28, 1>, Temp<29, 0>,
               Temp<30, 2>, Temp<31, 0>, Temp<32, -2>, Temp<33, 2>,
               Temp<34, -2>, Temp<35, -2>, Temp<36, -2>, Temp<37, 0>,
               Temp<38, 0>, Temp<39, 0>, Temp<40, 0>, Temp<41, 2>, Temp<42, 2>,
               Temp<43, 0>, Temp<44, 0>, Temp<45, 2>, Temp<46, 2>, Temp<47, 0>,
               Temp<48, 4>, Temp<49, 2>, Temp<51, 0>, Temp<52, 0>, Temp<53, 0>,
               Temp<54, 0>, Temp<55, 0>, Temp<56, 0>, Temp<57, 0>, Temp<58, 0>,
               Temp<59, 0>, Temp<60, 0>, Temp<61, 0>>;
}  // namespace

Scalar<SpinWeighted<ComplexDataVector, 2>> evaluate_worldtube_h_residual(
    const ComplexDataVector& dy_dy_j_value,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& j_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& w_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& q_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& h_numerical_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_h_numerical_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_scalar,
    const size_t l_max) {
  const size_t n = Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const auto& j = get(j_scalar);
  const auto& u = get(u_scalar);
  const auto& q = get(q_scalar);
  const auto& du_r = get(du_r_scalar);
  const auto& r = get(r_scalar);

  Variables<BufferTags> buffer{n};

  // ---- worldtube geometry from R (all y-independent) ----------------------
  // `one_minus_y` is 2 at the worldtube; it doubles as the pointwise spin-0
  // argument named `two` expected by the `ComputeBondiIntegrand` overloads.
  auto& one_minus_y = get<Temp<0, 0>>(buffer);
  get(one_minus_y).data() = std::complex<double>(2.0, 0.0);
  auto& two = one_minus_y;
  auto& eth_r_divided_by_r = get<Temp<1, 1>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_r_divided_by_r)), get(r_scalar));
  get(eth_r_divided_by_r) = get(eth_r_divided_by_r) / r;
  auto& eth_ethbar_r_divided_by_r = get<Temp<2, 0>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::EthEthbar>>(
      l_max, 1, make_not_null(&get(eth_ethbar_r_divided_by_r)), get(r_scalar));
  get(eth_ethbar_r_divided_by_r) = get(eth_ethbar_r_divided_by_r) / r;
  auto& eth_eth_r_divided_by_r = get<Temp<3, 2>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::EthEth>>(
      l_max, 1, make_not_null(&get(eth_eth_r_divided_by_r)), get(r_scalar));
  get(eth_eth_r_divided_by_r) = get(eth_eth_r_divided_by_r) / r;

  auto& k = get<Temp<4, 0>>(buffer);
  get(k) = sqrt(1.0 + j * conj(j));
  auto& exp_2_beta = get<Temp<5, 0>>(buffer);
  get(exp_2_beta) = exp(2.0 * get(beta_scalar));
  auto& du_r_divided_by_r = get<Temp<6, 0>>(buffer);
  get(du_r_divided_by_r) = du_r / r;

  // `dy_j`, `h_numerical` and `dy_h_numerical` are supplied directly in the
  // numerical (constant y) coordinate; the conversion from the physical
  // worldtube data is performed by the caller (`compute_dy_dy_j`).
  const auto& dy_j = get(dy_j_scalar);

  // y-independent products and the y-independent (dy^2 J-free) parts that the
  // H equation needs as operands.
  auto& dy_j_jbar = get(get<Temp<51, 0>>(buffer)).data();
  dy_j_jbar = dy_j.data() * conj(j.data()) + j.data() * conj(dy_j.data());
  auto& dy_jbar_dy_j = get(get<Temp<52, 0>>(buffer)).data();
  dy_jbar_dy_j = conj(dy_j.data()) * dy_j.data();
  auto& inverse_k_squared = get(get<Temp<53, 0>>(buffer)).data();
  inverse_k_squared = 1.0 / square(get(k).data());

  auto& dy_beta_scalar = get<Temp<7, 0>>(buffer);
  ComputeBondiIntegrand<Tags::Integrand<Tags::BondiBeta>>::apply(
      make_not_null(&dy_beta_scalar), dy_j_scalar, j_scalar, two);
  const ComplexDataVector& dy_beta = get(dy_beta_scalar).data();
  // dy^2 beta = (dy^2 beta with dy^2 J killed)
  //           + dy_dy_beta_coefficient * dy^2 J
  //           + dy_dy_beta_conjugate_coefficient * dy^2 Jbar,
  // obtained by differentiating the beta-integrand once more in y at the
  // worldtube.
  auto& dy_dy_beta_excluding = get(get<Temp<54, 0>>(buffer)).data();
  dy_dy_beta_excluding =
      -0.5 * dy_beta - dy_j_jbar * dy_beta * inverse_k_squared;
  auto& dy_dy_beta_coefficient = get(get<Temp<55, 0>>(buffer)).data();
  dy_dy_beta_coefficient =
      0.25 * (conj(dy_j.data()) -
              0.5 * dy_j_jbar * conj(j.data()) * inverse_k_squared);
  auto& dy_dy_beta_conjugate_coefficient = get(get<Temp<56, 0>>(buffer)).data();
  dy_dy_beta_conjugate_coefficient =
      0.25 * (dy_j.data() - 0.5 * dy_j_jbar * j.data() * inverse_k_squared);

  // ---- residual rhs - lhs of the H equation with dy^2 J = dy_dy_j_value ----
  const ComplexDataVector& ddj = dy_dy_j_value;
  auto& ddjbar = get(get<Temp<57, 0>>(buffer)).data();
  ddjbar = conj(ddj);
  // dy^2 of the relevant operands (with the dy^2 J pieces retained):
  auto& dy_dy_j_jbar = get(get<Temp<58, 0>>(buffer)).data();
  dy_dy_j_jbar = conj(j.data()) * ddj + 2.0 * dy_jbar_dy_j + j.data() * ddjbar;
  auto& dy_dy_beta = get(get<Temp<59, 0>>(buffer)).data();
  dy_dy_beta = dy_dy_beta_excluding + dy_dy_beta_coefficient * ddj +
               dy_dy_beta_conjugate_coefficient * ddjbar;

  // -- angular derivatives of J (numerical -> physical) --
  auto& ethbar_j = get<Temp<8, 1>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Ethbar>>(
      l_max, 1, make_not_null(&get(ethbar_j)), j);
  Jacobian<2, SwshTags::Ethbar>::apply(make_not_null(&ethbar_j), one_minus_y,
                                       eth_r_divided_by_r, dy_j.data());
  auto& eth_j_jbar = get<Temp<9, 1>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_j_jbar)), j * conj(j));
  Jacobian<0, SwshTags::Eth>::apply(make_not_null(&eth_j_jbar), one_minus_y,
                                    eth_r_divided_by_r, dy_j_jbar);
  auto& eth_jbar_dy_j = get<Temp<10, 1>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_jbar_dy_j)), conj(j) * dy_j);
  Jacobian<0, SwshTags::Eth>::apply(make_not_null(&eth_jbar_dy_j), one_minus_y,
                                    eth_r_divided_by_r,
                                    dy_jbar_dy_j + conj(j.data()) * ddj);
  auto& ethbar_dy_j = get<Temp<11, 1>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Ethbar>>(
      l_max, 1, make_not_null(&get(ethbar_dy_j)), dy_j);
  Jacobian<2, SwshTags::Ethbar>::apply(make_not_null(&ethbar_dy_j), one_minus_y,
                                       eth_r_divided_by_r, ddj);
  auto& ethbar_jbar_dy_j = get<Temp<12, -1>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Ethbar>>(
      l_max, 1, make_not_null(&get(ethbar_jbar_dy_j)), conj(j) * dy_j);
  Jacobian<0, SwshTags::Ethbar>::apply(make_not_null(&ethbar_jbar_dy_j),
                                       one_minus_y, eth_r_divided_by_r,
                                       dy_jbar_dy_j + conj(j.data()) * ddj);
  // physical eth/ethbar of dy_j feeding the second-derivative Jacobians
  auto& eth_dy_j = get<Temp<13, 3>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_dy_j)), dy_j);
  Jacobian<2, SwshTags::Eth>::apply(make_not_null(&eth_dy_j), one_minus_y,
                                    eth_r_divided_by_r, ddj);
  auto& eth_ethbar_j = get<Temp<14, 2>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::EthEthbar>>(
      l_max, 1, make_not_null(&get(eth_ethbar_j)), j);
  Jacobian<2, SwshTags::EthEthbar>::apply(
      make_not_null(&eth_ethbar_j), one_minus_y, eth_r_divided_by_r,
      eth_ethbar_r_divided_by_r, dy_j.data(), ddj, get(eth_dy_j).data(),
      get(ethbar_dy_j).data());
  auto& ethbar_ethbar_j = get<Temp<15, 0>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::EthbarEthbar>>(
      l_max, 1, make_not_null(&get(ethbar_ethbar_j)), j);
  Jacobian<2, SwshTags::EthbarEthbar>::apply(
      make_not_null(&ethbar_ethbar_j), one_minus_y, eth_r_divided_by_r,
      eth_eth_r_divided_by_r, dy_j.data(), ddj, get(ethbar_dy_j).data());
  // physical eth of dy(J Jbar) feeding the EthEthbar Jacobian of J Jbar
  auto& eth_dy_j_jbar = get<Temp<16, 1>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_dy_j_jbar)),
      SpinWeighted<ComplexDataVector, 0>{dy_j_jbar});
  Jacobian<0, SwshTags::Eth>::apply(make_not_null(&eth_dy_j_jbar), one_minus_y,
                                    eth_r_divided_by_r, dy_dy_j_jbar);
  auto& eth_ethbar_j_jbar = get<Temp<17, 0>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::EthEthbar>>(
      l_max, 1, make_not_null(&get(eth_ethbar_j_jbar)), j * conj(j));
  Jacobian<0, SwshTags::EthEthbar>::apply(
      make_not_null(&eth_ethbar_j_jbar), one_minus_y, eth_r_divided_by_r,
      eth_ethbar_r_divided_by_r, dy_j_jbar, dy_dy_j_jbar,
      get(eth_dy_j_jbar).data(), conj(get(eth_dy_j_jbar).data()));

  // -- derivatives of beta --
  auto& eth_beta = get<Temp<18, 1>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_beta)), get(beta_scalar));
  Jacobian<0, SwshTags::Eth>::apply(make_not_null(&eth_beta), one_minus_y,
                                    eth_r_divided_by_r, dy_beta);
  auto& eth_dy_beta = get<Temp<19, 1>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_dy_beta)), get(dy_beta_scalar));
  Jacobian<0, SwshTags::Eth>::apply(make_not_null(&eth_dy_beta), one_minus_y,
                                    eth_r_divided_by_r, dy_dy_beta);
  auto& eth_eth_beta = get<Temp<20, 2>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::EthEth>>(
      l_max, 1, make_not_null(&get(eth_eth_beta)), get(beta_scalar));
  Jacobian<0, SwshTags::EthEth>::apply(
      make_not_null(&eth_eth_beta), one_minus_y, eth_r_divided_by_r,
      eth_eth_r_divided_by_r, dy_beta, dy_dy_beta, get(eth_dy_beta).data());
  auto& eth_ethbar_beta = get<Temp<21, 0>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::EthEthbar>>(
      l_max, 1, make_not_null(&get(eth_ethbar_beta)), conj(get(beta_scalar)));
  Jacobian<0, SwshTags::EthEthbar>::apply(
      make_not_null(&eth_ethbar_beta), one_minus_y, eth_r_divided_by_r,
      eth_ethbar_r_divided_by_r, dy_beta, dy_dy_beta, get(eth_dy_beta).data(),
      conj(get(eth_dy_beta).data()));

  // -- derivatives of Q --
  auto& dy_q = get<Temp<22, 1>>(buffer);
  {
    auto& pole_q = get<Temp<23, 1>>(buffer);
    auto& regular_q = get<Temp<24, 1>>(buffer);
    auto& script_aq = get<Temp<25, 1>>(buffer);
    ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiQ>>::apply(
        make_not_null(&pole_q), eth_beta);
    ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiQ>>::apply(
        make_not_null(&regular_q), make_not_null(&script_aq), dy_beta_scalar,
        dy_j_scalar, j_scalar, eth_dy_beta, eth_j_jbar, eth_jbar_dy_j,
        ethbar_dy_j, ethbar_j, eth_r_divided_by_r, k);
    get(dy_q) = 0.5 * get(pole_q) + get(regular_q) - q;
  }
  auto& eth_q = get<Temp<26, 2>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_q)), q);
  Jacobian<1, SwshTags::Eth>::apply(make_not_null(&eth_q), one_minus_y,
                                    eth_r_divided_by_r, get(dy_q).data());
  auto& ethbar_q = get<Temp<27, 0>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Ethbar>>(
      l_max, 1, make_not_null(&get(ethbar_q)), q);
  Jacobian<1, SwshTags::Ethbar>::apply(make_not_null(&ethbar_q), one_minus_y,
                                       eth_r_divided_by_r, get(dy_q).data());

  // -- derivatives of U --
  auto& dy_u = get<Temp<28, 1>>(buffer);
  ComputeBondiIntegrand<Tags::Integrand<Tags::BondiU>>::apply(
      make_not_null(&dy_u), exp_2_beta, j_scalar, q_scalar, k, r_scalar);
  auto& dy_k = get(get<Temp<60, 0>>(buffer)).data();
  dy_k = (dy_j.data() * conj(j.data()) + j.data() * conj(dy_j.data())) /
         (2.0 * get(k).data());
  auto& dy_dy_u = get(get<Temp<61, 0>>(buffer)).data();
  dy_dy_u =
      2.0 * dy_beta * get(dy_u).data() +
      get(exp_2_beta).data() *
          (get(dy_q).data() * get(k).data() + q.data() * dy_k -
           dy_j.data() * conj(q.data()) - j.data() * conj(get(dy_q).data())) /
          (2.0 * r.data());
  auto& ethbar_dy_u = get<Temp<29, 0>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Ethbar>>(
      l_max, 1, make_not_null(&get(ethbar_dy_u)), get(dy_u));
  Jacobian<1, SwshTags::Ethbar>::apply(make_not_null(&ethbar_dy_u), one_minus_y,
                                       eth_r_divided_by_r, dy_dy_u);
  auto& eth_u = get<Temp<30, 2>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_u)), u);
  Jacobian<1, SwshTags::Eth>::apply(make_not_null(&eth_u), one_minus_y,
                                    eth_r_divided_by_r, get(dy_u).data());
  auto& ethbar_u = get<Temp<31, 0>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Ethbar>>(
      l_max, 1, make_not_null(&get(ethbar_u)), u);
  Jacobian<1, SwshTags::Ethbar>::apply(make_not_null(&ethbar_u), one_minus_y,
                                       eth_r_divided_by_r, get(dy_u).data());

  // -- extra product derivatives --
  auto& ethbar_jbar_u = get<Temp<32, -2>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Ethbar>>(
      l_max, 1, make_not_null(&get(ethbar_jbar_u)), conj(j) * u);
  Jacobian<-1, SwshTags::Ethbar>::apply(
      make_not_null(&ethbar_jbar_u), one_minus_y, eth_r_divided_by_r,
      conj(j.data()) * get(dy_u).data() + u.data() * conj(dy_j.data()));
  auto& eth_ubar_dy_j = get<Temp<33, 2>>(buffer);
  Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Eth>>(
      l_max, 1, make_not_null(&get(eth_ubar_dy_j)), conj(u) * dy_j);
  Jacobian<1, SwshTags::Eth>::apply(
      make_not_null(&eth_ubar_dy_j), one_minus_y, eth_r_divided_by_r,
      conj(get(dy_u).data()) * dy_j.data() + conj(u.data()) * ddj);
  auto& ethbar_jbar_q_minus_2_eth_beta = get<Temp<34, -2>>(buffer);
  {
    auto& ethbar_jbar_q = get<Temp<35, -2>>(buffer);
    Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Ethbar>>(
        l_max, 1, make_not_null(&get(ethbar_jbar_q)), conj(j) * q);
    Jacobian<-1, SwshTags::Ethbar>::apply(
        make_not_null(&ethbar_jbar_q), one_minus_y, eth_r_divided_by_r,
        conj(dy_j.data()) * q.data() + conj(j.data()) * get(dy_q).data());
    auto& ethbar_jbar_eth_beta = get<Temp<36, -2>>(buffer);
    Spectral::Swsh::angular_derivatives<tmpl::list<SwshTags::Ethbar>>(
        l_max, 1, make_not_null(&get(ethbar_jbar_eth_beta)),
        conj(j) * get(eth_beta));
    // The operand here is dy(Jbar eth beta), where eth beta is the physical
    // (constant r) angular derivative. Since dy does not commute with the
    // physical eth, dy(eth beta) = eth(dy beta) + (eth R / R) dy beta picks up
    // the worldtube-radius gradient term, which matters for an angle-dependent
    // worldtube radius.
    Jacobian<-1, SwshTags::Ethbar>::apply(
        make_not_null(&ethbar_jbar_eth_beta), one_minus_y, eth_r_divided_by_r,
        conj(dy_j.data()) * get(eth_beta).data() +
            conj(j.data()) * (get(eth_dy_beta).data() +
                              get(eth_r_divided_by_r).data() * dy_beta));
    get(ethbar_jbar_q_minus_2_eth_beta) =
        get(ethbar_jbar_q) - 2.0 * get(ethbar_jbar_eth_beta);
  }

  // -- dy W from the W-hypersurface equation --
  auto& dy_w = get<Temp<37, 0>>(buffer);
  {
    auto& pole_w = get<Temp<38, 0>>(buffer);
    auto& regular_w = get<Temp<39, 0>>(buffer);
    auto& script_av = get<Temp<40, 0>>(buffer);
    ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiW>>::apply(
        make_not_null(&pole_w), ethbar_u);
    ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiW>>::apply(
        make_not_null(&regular_w), make_not_null(&script_av), dy_u, exp_2_beta,
        j_scalar, q_scalar, eth_beta, eth_eth_beta, eth_ethbar_beta,
        eth_ethbar_j, eth_ethbar_j_jbar, eth_j_jbar, ethbar_dy_u,
        ethbar_ethbar_j, ethbar_j, eth_r_divided_by_r, k, r_scalar);
    get(dy_w) = 0.5 * get(pole_w) + get(regular_w) - get(w_scalar);
  }

  // -- right- and left-hand sides of the H-hypersurface equation --
  auto& pole_h = get<Temp<41, 2>>(buffer);
  ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiH>>::apply(
      make_not_null(&pole_h), j_scalar, u_scalar, w_scalar, eth_u, ethbar_j,
      ethbar_jbar_u, ethbar_u, k);
  auto& regular_h = get<Temp<42, 2>>(buffer);
  auto& script_aj = get<Temp<43, 0>>(buffer);
  auto& script_bj = get<Temp<44, 0>>(buffer);
  auto& script_cj = get<Temp<45, 2>>(buffer);
  auto& dy_dy_j_argument = get<Temp<46, 2>>(buffer);
  get(dy_dy_j_argument).data() = ddj;
  ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiH>>::apply(
      make_not_null(&regular_h), make_not_null(&script_aj),
      make_not_null(&script_bj), make_not_null(&script_cj), dy_dy_j_argument,
      dy_j_scalar, dy_w, exp_2_beta, j_scalar, q_scalar, u_scalar, w_scalar,
      eth_beta, eth_eth_beta, eth_ethbar_beta, eth_ethbar_j, eth_ethbar_j_jbar,
      eth_j_jbar, eth_q, eth_u, eth_ubar_dy_j, ethbar_dy_j, ethbar_ethbar_j,
      ethbar_j, ethbar_jbar_dy_j, ethbar_jbar_q_minus_2_eth_beta, ethbar_q,
      ethbar_u, du_r_divided_by_r, eth_r_divided_by_r, k, two, r_scalar);

  auto& linear_factor = get<Temp<47, 0>>(buffer);
  auto& linear_factor_conjugate = get<Temp<48, 4>>(buffer);
  auto& script_djbar = get<Temp<49, 2>>(buffer);
  ComputeBondiIntegrand<Tags::LinearFactor<Tags::BondiH>>::apply(
      make_not_null(&linear_factor), make_not_null(&script_djbar), dy_j_scalar,
      j_scalar, two);
  ComputeBondiIntegrand<Tags::LinearFactorForConjugate<Tags::BondiH>>::apply(
      make_not_null(&linear_factor_conjugate), make_not_null(&script_djbar),
      dy_j_scalar, j_scalar, two);

  Scalar<SpinWeighted<ComplexDataVector, 2>> residual{n};
  get(residual).data() =
      get(pole_h).data() + 2.0 * get(regular_h).data() -
      (2.0 * get(dy_h_numerical_scalar).data() +
       get(linear_factor).data() * get(h_numerical_scalar).data() +
       get(linear_factor_conjugate).data() *
           conj(get(h_numerical_scalar).data()));
  return residual;
}

void compute_dy_dy_j(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dy_dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& j_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& u_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& w_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& q_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& h_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& du_dr_j_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r_scalar,
    const size_t l_max) {
  const size_t n = Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // Convert the physical worldtube data to the numerical (constant y)
  // coordinate that `evaluate_worldtube_h_residual` works in, using the
  // worldtube Jacobian dy_j = (R / 2) Dr<J>:
  //   dy_j           = (R / 2) Dr<J>,
  //   h_numerical    = H + Du<R> Dr<J>            (= Du<J> at constant y),
  //   dy_h_numerical = (1 / 2) (Du<R> Dr<J> + R Du<Dr<J>>)
  //                                               (= Dy of h_numerical).
  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_j_scalar{n};
  get(dy_j_scalar).data() =
      0.5 * get(r_scalar).data() * get(dr_j_scalar).data();
  Scalar<SpinWeighted<ComplexDataVector, 2>> h_numerical_scalar{n};
  get(h_numerical_scalar).data() =
      get(h_scalar).data() + get(du_r_scalar).data() * get(dr_j_scalar).data();
  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_h_numerical_scalar{n};
  get(dy_h_numerical_scalar).data() =
      0.5 * (get(du_r_scalar).data() * get(dr_j_scalar).data() +
             get(r_scalar).data() * get(du_dr_j_scalar).data());
  const auto residual = [&](const ComplexDataVector& dy_dy_j_value) {
    return get(evaluate_worldtube_h_residual(
                   dy_dy_j_value, j_scalar, u_scalar, w_scalar, beta_scalar,
                   q_scalar, dy_j_scalar, h_numerical_scalar,
                   dy_h_numerical_scalar, du_r_scalar, r_scalar, l_max))
        .data();
  };

  // ---- extract c1, c2, c3 by probing the affine residual, then solve -------
  // residual(X) = c1 X + c2 Xbar + c3, so
  //   c3 = residual(0),  c1 + c2 = residual(1) - c3,
  //   c1 - c2 = -i(residual(i) - c3).
  const ComplexDataVector c3 = residual(ComplexDataVector{n, 0.0});
  const ComplexDataVector sum = residual(ComplexDataVector{n, 1.0}) - c3;
  const ComplexDataVector difference =
      std::complex<double>(0.0, -1.0) *
      (residual(ComplexDataVector{n, std::complex<double>(0.0, 1.0)}) - c3);
  const ComplexDataVector c1 = 0.5 * (sum + difference);
  const ComplexDataVector c2 = 0.5 * (sum - difference);
  // Solve c1 dy^2 J + c2 dy^2 Jbar = -c3 using the conjugate pair.
  get(*dy_dy_j).data() =
      (conj(c1) * c3 - c2 * conj(c3)) / (c2 * conj(c2) - c1 * conj(c1));
}

}  // namespace Cce::InitializeJ::CauchySecondOrder_detail
