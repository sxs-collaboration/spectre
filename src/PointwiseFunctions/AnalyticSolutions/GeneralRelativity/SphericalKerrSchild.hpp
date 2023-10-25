// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace gr {
namespace Solutions {

/*!
 * \brief Kerr black hole in Spherical Kerr-Schild coordinates
 *
 * ## Introduction
 *
 * Given a Kerr-Schild (KS) black hole system, we denote the coordinate system
 * using \f$\{t,x,y,z\}\f$. In the transformed system, Spherical Kerr-Schild
 * (Spherical KS), we denote the coordinate system using
 * \f$\{t,\bar{x},\bar{y},\bar{z}\}\f$. Further, when considering indexed
 * objects, we will use Greek and Latin indices with the standard convention
 * \f$(\mu=0,1,2,3\f$ and \f$i=1,2,3\f$ respectively), but we will use a bar to
 * denote that a given index is in reference to the Spherical KS coordinate
 * system (i.e. \f$i\f$ vs. \f$\bar{\imath}\f$ and \f$\mu\f$ vs.
 * \f$\bar{\mu}\f$).
 *
 * ## Spin in the z direction
 *
 * ### The Transformation
 *
 * The Boyer-Lindquist radius for KS with its spin in the \f$z\f$ direction is
 * defined by
 *
 * \f{align}{
 *    \frac{x^2 + y^2}{r^2 + a^2} + \frac{z^2}{r^2} = 1,
 * \f}
 *
 * or equivalently,
 *
 * \f{align}{
 *    r^2 &= \frac{1}{2}\left(x^2+y^2+z^2-a^2\right) +
 *    \left(\frac{1}{4}\left(x^2+y^2+z^2-a^2\right)^2 + a^2z^2\right)^{1/2}.
 * \f}
 *
 * The Spherical KS coordinates
 *
 * \f{align}{
 *    \vec{\bar{x}} = x^{\bar{\imath}} = x_{\bar{\imath}} =
 *    (\bar{x},\bar{y},\bar{z}),
 * \f}
 *
 * and KS coordinates
 *
 * \f{align}{
 *    \vec{x} = x^{i} = x_{i} = (x,y,z),
 * \f}
 *
 * are related by
 *
 * \f{align}{
 *    \left(\frac{\bar{x}}{r},\frac{\bar{y}}{r},\frac{\bar{z}}{r}\right) \equiv
 *    \left(\frac{x}{\rho},\frac{y}{\rho},\frac{z}{r}\right),
 * \f}
 *
 * where we have defined
 *
 * \f{align}{
 *   \rho^2 \equiv r^2 + a^2. \label{eq:rho}
 * \f}
 *
 * Therefore, we have that \f$r\f$ satisfies the equation for a sphere in
 * Spherical KS
 *
 * \f{align}{
 *    r^2 = \vec{\bar{x}}\cdot\vec{\bar{x}} = \bar{x}^2 + \bar{y}^2 +
 *    \bar{z}^2.
 * \f}
 *
 * It is clear to see that the Spherical KS radius coincides with the
 * Boyer-Lindquist radius.
 *
 * ## Spin in an Arbitrary Direction
 *
 * Given that the remaining important quantities take forms that are easily
 * specialized to the \f$z\f$ spin case, we instead focus on the general case
 * for brevity.
 *
 * ### The Transformation
 *
 * The Boyer-Lindquist radius for KS with spin in an arbitrary direction is
 * defined by
 *
 * \f{align}{
 *    r^2 = \frac{1}{2}\left(\vec{x}\cdot\vec{x}-a^2\right) +
 *    \left(\frac{1}{4}\left(\vec{x}\cdot\vec{x}-a^2\right)^2 +
 *    \left(\vec{a}\cdot\vec{x}\right)^2\right)^{1/2}.
 * \f}
 *
 * Then, defining two transformation matrices \f$Q^{\bar{\imath}}{}_{j}\f$ and
 * \f$P^{j}{}_{\bar{\imath}}\f$ as
 *
 * \f{align}{
 *    Q^{\bar{\imath}}{}_{j} &= \frac{r}{\rho}\delta^{\bar{\imath}}{}_{j} +
 *    \frac{1}{(\rho + r)\rho}a^{\bar{\imath}}a_{j}, \\
 *    P^{j}{}_{\bar{\imath}} &= \frac{\rho}{r}\delta^{j}{}_{\bar{\imath}} -
 *    \frac{1}{(\rho + r)r}a^{j}a_{\bar{\imath}},
 * \f}
 *
 * where the definition of \f$\rho\f$ is identical to Eq. \f$(\ref{eq:rho})\f$,
 * such that
 *
 * \f{align}{
 *   Q^{\bar{\imath}}{}_{j}x^{j} &= x^{\bar{\imath}}, \\
 *   P^{j}{}_{\bar{\imath}}x^{\bar{\imath}} &= x^{j}.
 * \f}
 *
 * We again recover that \f$r\f$ satisfies the equation for a sphere in
 * Spherical KS
 *
 * \f{align}{
 *   r^2 = \vec{\bar{x}}\cdot\vec{\bar{x}} = \bar{x}^2+\bar{y}^2+\bar{z}^2,
 * \f}
 *
 * and we also have that
 *
 * \f{align}{
 *   \vec{a}\cdot\vec{\bar{x}} = \vec{a}\cdot\vec{x}.
 * \f}
 *
 * Note that \f$Q^{\bar{\imath}}{}_{ j}\f$ and \f$P^{j}{}_{\bar{\imath}}\f$
 * satisfy
 *
 * \f{align}{
 *   P^{i}{}_{\bar{\jmath}}\,Q^{\bar{\jmath}}{}_{ k} = \delta^{i}{}_{ k}.
 * \f}
 *
 * The Jacobian is then given by
 *
 * \f{align}{
 *   T^{i}{}_{\bar{\jmath}} = \partial_{\bar{\jmath}}\,x^{i} =
 *   \frac{\partial x^i}{\partial x^{\bar{\jmath}}},
 * \f}
 *
 * while its inverse is given by
 *
 * \f{align}{
 *   S^{\bar{\jmath}}{}_{ i} = \partial_{i}\,x^{\bar{\jmath}} =
 *   \frac{\partial x^{\bar{\jmath}}}{\partial x^i},
 * \f}
 *
 * which in turn satisfies
 *
 * \f{align}{
 *   T^{i}{}_{\bar{\jmath}}\,S^{\bar{\jmath}}{}_{ k} &= \delta^i_{\; k}.
 * \f}
 *
 * ### The Metric
 *
 * A KS coordinate system is defined by
 *
 * \f{align}{
 *   g_{\mu\nu} = \eta_{\mu\nu} + 2Hl_{\mu}l_{\nu},
 * \f}
 *
 * where \f$H\f$ is a scalar function of the coordinates, \f$\eta_{\mu\nu}\f$ is
 * the Minkowski metric, and \f$l^\mu\f$ is a null vector. Note that the inverse
 * of the spacetime metric is given by
 *
 * \f{align}{
 *   g^{\mu\nu} = \eta^{\mu\nu} - 2Hl^{\mu}l^{\nu}.
 * \f}
 *
 * The scalar function \f$H\f$ takes the form
 *
 * \f{align}{
 *   H = \frac{Mr^3}{r^4 + \left(\vec{a}\cdot\vec{x}\right)^{2}},
 * \f}
 *
 * where \f$M\f$ is the mass, while the spatial part of the null vector takes
 * the form
 *
 * \f{align}{
 *   l_{i} = l^{i} = \frac{r\vec{x} - \vec{a}\times\vec{x} +
 *   \frac{(\vec{a}\cdot\vec{x})\vec{a}}{r}}{\rho^2}.
 * \f}
 *
 * Note that the full spacetime form of the null vector is
 *
 * \f{align}{
 *   l_{\mu} &= (-1,l_{i}), & l^{\mu} &= (1,l^{i}).
 * \f}
 *
 * Transforming the KS spatial metric then yields the following Spherical KS
 * spatial metric
 *
 * \f{align}{
 *   \gamma_{\bar{\imath}\bar{\jmath}} &=
 *   \gamma_{mn}T^{m}{}_{\bar{\imath}}\,T^{n}{}_{\bar{\jmath}}, \nonumber \\
 *   &= \eta_{mn}T^{m}{}_{\bar{\imath}}\,T^{n}{}_{\bar{\jmath}} +
 *   2Hl_{m}l_{n}T^{m}{}_{\bar{\imath}}\,T^{n}{}_{\bar{\jmath}}, \nonumber \\
 *   &= \eta_{\bar{\imath}\bar{\jmath}} + 2Hl_{\bar{\imath}}l_{\bar{\jmath}}.
 * \f}
 *
 * The transformed spacetime Minkowski metric is given by
 *
 * \f{align}{
 *   \eta_{\bar{\mu}\bar{\nu}} = (-1)\otimes\eta_{\bar{\imath}\bar{\jmath}},
 * \f}
 *
 * and the transformed spacetime null vector is given by
 *
 * \f{align}{
 *   l_{\bar{\mu}} &= (-1,l_{\bar{\imath}}), &
 *   l^{\bar{\mu}} &= (1,l^{\bar{\imath}}).
 * \f}
 *
 * Therefore, the Spherical KS spacetime metric is
 *
 * \f{align}{
 *   g_{\bar{\mu}\bar{\nu}} = \eta_{\bar{\mu}\bar{\nu}} +
 *   2Hl_{\bar{\mu}}l_{\bar{\nu}}.
 * \f}
 *
 * Further, we have that the lapse in Spherical KS is given by
 *
 * \f{align}{
 *   \alpha = \left(1 + 2H\right)^{-1/2},
 * \f}
 *
 * and the shift in Spherical KS by
 *
 * \f{align}{
 *   \beta^{\bar{\imath}} &= -\frac{2Hl^{t}l^{\bar{\imath}}}{1 + 2Hl^{t}l^{t}} =
 *   -2H\alpha^{2}l^{t}l^{\bar{\imath}}, & \beta_{\bar{\imath}} &=
 *  -2Hl_{t}l_{\bar{\imath}}.
 * \f}
 *
 * ### Derivatives
 *
 * The derivatives of the preceding quantities are
 *
 * \f{align}{
 *   \frac{\partial r}{\partial x^{i}} &= \frac{r^{2}x_{i} +
 *   \left(\vec{a}\cdot\vec{x}\right)a_{i}}{rs}, \\
 *   \partial_{\bar{\imath}}H &= HT^{m}{}_{\bar{\imath}}
 *   \left[\frac{3}{r}\frac{\partial r}{\partial x^{m}} -
 *   \frac{4r^{3}\frac{\partial r}{\partial x^{m}} +
 *   2\left(\vec{a}\cdot\vec{x}\right)a_{m}}{r^{4} +
 *   \left(\vec{a}\cdot\vec{x}\right)^{2}}\right], \\
 *   \partial_{\bar{\jmath}}l^{\bar{\imath}} &=
 *   \partial_{\bar{\jmath}}\left(l_{k}T^{k}{}_{\bar{\imath}}\right),
 *   \nonumber \\
 *   &= T^{k}{}_{\bar{\imath}}T^{m}{}_{\bar{\jmath}}\frac{1}{\rho^{2}}
 *   \left[\left(x_{k} - 2rl_{k} -
 *   \frac{\left(\vec{a}\cdot\vec{x}\right)a_{k}}{r^{2}}\right)
 *   \frac{\partial r}{\partial x^{m}} + r\delta_{km} + \frac{a_{k}a_{m}}{r} +
 *   \epsilon^{kmn}a_{n}\right] +
 *   l_{k}\partial_{\bar{\jmath}}T^{k}{}_{\bar{\imath}}, \\
 *   \partial_{\bar{k}}\gamma_{\bar{\imath}\bar{\jmath}} &=
 *   2l_{\bar{\imath}}l_{\bar{\jmath}}\partial_{\bar{k}}H +
 *   4Hl_{(\bar{\imath}}\partial_{\bar{k}}l_{\bar{\jmath})} +
 *   T^{m}{}_{\bar{\jmath}}\partial_{\bar{k}}T^{m}{}_{\bar{\imath}} +
 *   T^{m}{}_{\bar{\imath}}\partial_{\bar{k}}T^{m}{}_{\bar{\jmath}}, \\
 *   \partial_{\bar{k}}\alpha &= -\left(1+2H\right)^{-3/2}\partial_{\bar{k}}H =
 *   -\alpha^{3}\partial_{\bar{k}}H, \\
 *   \partial_{\bar{k}}\beta^{i} &=
 *   2\alpha^{2}\left[l^{\bar{\imath}}\partial_{\bar{k}}H +
 *   H\left(S^{\bar{\imath}}{}_{j}S^{\bar{m}}{}_{n}
 *   \delta_{nj}\partial_{\bar{k}}l_{\bar{m}} +
 *   S^{\bar{\imath}}{}_{j}l_{\bar{m}}\partial_{\bar{k}}S^{\bar{m}}{}_{n}
 *   \delta_{nj} + S^{\bar{m}}{}_{n}\delta_{nj}l_{\bar{m}}
 *   \partial_{\bar{k}}S^{\bar{\imath}}{}_{j}\right)\right] -
 *   4Hl^{\bar{\imath}}\alpha^{4}\partial_{\bar{k}}H,
 * \f}
 *
 * where we have defined \f$s\f$ as
 *
 * \f{align}{
 *   s &\equiv r^{2} + \frac{\left(\vec{a}\cdot\vec{x}\right)^{2}}{r^{2}}.
 *   \label{eq: s_number}
 * \f}
 *
 * ## Code
 *
 * While the previous sections described the relevant physical quantities, the
 * actual files make use of internally defined objects to ease the computation
 * of the Jacobian, the inverse Jacobian, and their corresponding derivatives.
 * Therefore, we now list these intermediary objects as well as how they
 * construct the aforementioned physical quantities with the appropriate
 * definition found in the code (all for arbitrary spin).
 *
 * ### Helper Matrices
 *
 * The intermediary objects used to define the various Jacobian objects, the
 * so-called "helper matrices", are defined below.
 *
 * \f{align}{
 *   F^{i}{}_{\bar{k}} &\equiv
 *   -\frac{1}{\rho r^{3}}\left(a^{2}\delta^{i}{}_{\bar{k}} -
 *   a^{i}a_{\bar{k}}\right), \\
 *   \left(G_1\right)^{\bar{\imath}}{}_{ \bar{m}} &\equiv
 *   \frac{1}{\rho^{2}r}\left(a^{2}\delta^{\bar{\imath}}{}_{\bar{m}} -
 *   a^{\bar{\imath}}a_{\bar{m}}\right), \\
 *   \left(G_2\right)^{\bar{n}}{}_{ j} &\equiv
 *   \frac{\rho^{2}}{sr}Q^{\bar{n}}{}_{j}, \\
 *   D^{i}{}_{\bar{m}} &\equiv
 *   \frac{1}{\rho^{3}r}\left(a^{2}\delta^{i}{}_{\bar{m}} -
 *   a^{i}a_{\bar{m}}\right), \\
 *   C^{i}{}_{\bar{m}} &\equiv D^{i}{}_{\bar{m}} - 3F^{i}{}_{\bar{m}} =
 *   \frac{1}{\rho r}\left(\frac{1}{\rho^{2}} +
 *   \frac{3}{r^{2}}\right)\left(a^{2}\delta^{i}{}_{\bar{m}} -
 *   a^{i}a_{\bar{m}}\right), \\
 *   \left(E_1\right)^{i}{}_{\bar{m}} &\equiv
 *   -\frac{1}{\rho^{2}}\left(\frac{1}{r^{2}} +
 *   \frac{2}{\rho^{2}}\right)\left(a^{2}\delta^{i}{}_{\bar{m}} -
 *   a^{i}a_{\bar{m}}\right), \nonumber \\
 *   &= -\frac{\left(\rho^{2} +
 *   2r^{2}\right)}{r^{2}\rho^{4}}\left(a^{2}\delta^{i}{}_{\bar{m}} -
 *   a^{i}a_{\bar{m}}\right), \\
 *   \left(E_2\right)^{\bar{n}}{}_{ j} &\equiv \left[-\frac{a^{2}}{\rho^{2}r} -
 *   \frac{2}{s}\left(r - \frac{\left(\vec{a}\cdot\vec{x}\right)^{2}}{r^{3}}
 *   \right)\right]\cdot\left(G_2\right)^{\bar{n}}{}_{ j} +
 *   \frac{1}{s}P^{\bar{n}}{}_{ j},
 * \f}
 *
 * where \f$s\f$ is defined identically to Eq. \f$(\ref{eq: s_number})\f$.
 *
 * ### Physical Quantities
 *
 * Below are the definitions for how we construct the Jacobian, inverse
 * Jacobian, derivative of the Jacobian, and derivative of the inverse Jacobian
 * in the code using the helper matrices.
 *
 * \f{align}{
 *   T^{i}{}_{\bar{\jmath}} &= P^{i}{}_{\bar{\jmath}} +
 *   F^{i}{}_{\bar{k}}x^{\bar{k}}x_{\bar{\jmath}}, \\
 *   S^{\bar{\imath}}{}_{ j} &= Q^{\bar{\imath}}{}_{ j} +
 *   \left(G_1\right)^{\bar{\imath}}{}_{\bar{m}}x^{\bar{m}}x_{\bar{n}}
 *   \left(G_2\right)^{\bar{n}}{}_{ j}, \\
 *   \partial_{\bar{k}}T^{i}{}_{\bar{\jmath}} &=
 *   F^{i}{}_{\bar{\jmath}}x_{\bar{k}} + F^{i}{}_{\bar{k}}x_{\bar{\jmath}} +
 *   F^{i}{}_{\bar{m}}x^{\bar{m}}\delta_{jk} +
 *   C^{i}{}_{\bar{m}}\frac{x_{\bar{k}}x^{\bar{m}}x_{\bar{\jmath}}}{r^{2}}, \\
 *   \partial_{\bar{k}}S^{\;\bar{\imath}}{}_{ j} &=
 *   D^{\bar{\imath}}{}_{j}x_{\bar{k}} +
 *   \left(G_1\right)^{\bar{\imath}}{}_{\bar{k}}x_{\bar{n}}
 *   \left(G_2\right)^{\bar{n}}{}_{j} +
 *   \left(G_1\right)^{\bar{\imath}}{}_{\bar{m}}x^{\bar{m}}
 *   \left(G_2\right)^{\bar{n}}{}_{j}\delta_{\bar{n}\bar{k}} \nonumber \\
 *   &\quad +
 *   \left(E_1\right)^{i}{}_{\bar{m}}\frac{x_{\bar{k}}x^{\bar{m}}x_{\bar{n}}}{r}
 *   \left(G_2\right)^{\bar{n}}{}_{j} +
 *   \left(G_1\right)^{\bar{i}}{}_{\bar{m}}
 *   \frac{x_{\bar{k}}x^{\bar{m}}x_{\bar{n}}}{r}\left(E_2\right)^{\bar{n}}{}_{j}
 *   - \left(G_1\right)^{\bar{\imath}}{}_{\bar{m}}x^{\bar{m}}x_{\bar{n}}
 *   \left(G_2\right)^{\bar{n}}{}_{j}
 *   \frac{2\left(\vec{a}\cdot\vec{x}\right)}{sr^{2}}a_{\bar{k}}.
 * \f}
 */
class SphericalKerrSchild : public AnalyticSolution<3_st>,
                            public MarkAsAnalyticSolution {
 public:
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"Mass of the black hole"};
    static type lower_bound() { return 0.; }
  };
  struct Spin {
    using type = std::array<double, volume_dim>;
    static constexpr Options::String help = {
        "The [x,y,z] dimensionless spin of the black hole"};
  };
  struct Center {
    using type = std::array<double, volume_dim>;
    static constexpr Options::String help = {
        "The [x,y,z] center of the black hole"};
  };
  using options = tmpl::list<Mass, Spin, Center>;
  static constexpr Options::String help{
      "Black hole in Spherical Kerr-Schild coordinates"};

  template <typename DataType, typename Frame = Frame::Inertial>
  using tags = tmpl::flatten<tmpl::list<
      AnalyticSolution<3_st>::tags<DataType, Frame>,
      gr::Tags::DerivDetSpatialMetric<DataType, 3, Frame>,
      gr::Tags::TraceExtrinsicCurvature<DataType>,
      gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame>,
      gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>,
      gr::Tags::TraceSpatialChristoffelSecondKind<DataType, 3, Frame>>>;

  SphericalKerrSchild(double mass, Spin::type dimensionless_spin,
                      Center::type center,
                      const Options::Context& context = {});

  explicit SphericalKerrSchild(CkMigrateMessage* /*unused*/);

  SphericalKerrSchild() = default;
  SphericalKerrSchild(const SphericalKerrSchild& /*rhs*/) = default;
  SphericalKerrSchild& operator=(const SphericalKerrSchild& /*rhs*/) = default;
  SphericalKerrSchild(SphericalKerrSchild&& /*rhs*/) = default;
  SphericalKerrSchild& operator=(SphericalKerrSchild&& /*rhs*/) = default;
  ~SphericalKerrSchild() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  SPECTRE_ALWAYS_INLINE double mass() const { return mass_; }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>& center() const {
    return center_;
  }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>&
  dimensionless_spin() const {
    return dimensionless_spin_;
  }

  struct internal_tags {
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_minus_center = ::Tags::TempI<0, 3, Frame, DataType>;
    template <typename DataType>
    using r_squared = ::Tags::TempScalar<1, DataType>;
    template <typename DataType>
    using r = ::Tags::TempScalar<2, DataType>;
    template <typename DataType>
    using rho = ::Tags::TempScalar<3, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using helper_matrix_F = ::Tags::TempIj<4, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using transformation_matrix_P = ::Tags::TempIj<5, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using jacobian = ::Tags::TempIj<6, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using helper_matrix_D = ::Tags::TempIj<7, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using helper_matrix_C = ::Tags::TempIj<8, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_jacobian = ::Tags::TempiJk<9, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using transformation_matrix_Q = ::Tags::TempIj<10, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using helper_matrix_G1 = ::Tags::TempIj<11, 3, Frame, DataType>;
    template <typename DataType>
    using a_dot_x = ::Tags::TempScalar<12, DataType>;
    template <typename DataType>
    using s_number = ::Tags::TempScalar<13, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using helper_matrix_G2 = ::Tags::TempIj<14, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using G1_dot_x = ::Tags::TempI<15, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using G2_dot_x = ::Tags::Tempi<16, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using inv_jacobian = ::Tags::TempIj<17, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using helper_matrix_E1 = ::Tags::TempIj<18, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using helper_matrix_E2 = ::Tags::TempIj<19, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_inv_jacobian = ::Tags::TempiJk<20, 3, Frame, DataType>;
    template <typename DataType>
    using H = ::Tags::TempScalar<21, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using kerr_schild_x = ::Tags::TempI<22, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using a_cross_x = ::Tags::TempI<23, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using kerr_schild_l = ::Tags::TempI<24, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using l_lower = ::Tags::Tempi<25, 4, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using l_upper = ::Tags::TempI<26, 4, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_r = ::Tags::TempI<27, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_H = ::Tags::TempI<28, 4, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using kerr_schild_deriv_l = ::Tags::Tempij<29, 4, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_l = ::Tags::Tempij<30, 4, Frame, DataType>;
    template <typename DataType>
    using lapse_squared = ::Tags::TempScalar<31, DataType>;
    template <typename DataType>
    using deriv_lapse_multiplier = ::Tags::TempScalar<32, DataType>;
    template <typename DataType>
    using shift_multiplier = ::Tags::TempScalar<33, DataType>;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  using CachedBuffer = CachedTempBuffer<
      internal_tags::x_minus_center<DataType, Frame>,
      internal_tags::r_squared<DataType>, internal_tags::r<DataType>,
      internal_tags::rho<DataType>,
      internal_tags::helper_matrix_F<DataType, Frame>,
      internal_tags::transformation_matrix_P<DataType, Frame>,
      internal_tags::jacobian<DataType, Frame>,
      internal_tags::helper_matrix_D<DataType, Frame>,
      internal_tags::helper_matrix_C<DataType, Frame>,
      internal_tags::deriv_jacobian<DataType, Frame>,
      internal_tags::transformation_matrix_Q<DataType, Frame>,
      internal_tags::helper_matrix_G1<DataType, Frame>,
      internal_tags::a_dot_x<DataType>, internal_tags::s_number<DataType>,
      internal_tags::helper_matrix_G2<DataType, Frame>,
      internal_tags::G1_dot_x<DataType, Frame>,
      internal_tags::G2_dot_x<DataType, Frame>,
      internal_tags::inv_jacobian<DataType, Frame>,
      internal_tags::helper_matrix_E1<DataType, Frame>,
      internal_tags::helper_matrix_E2<DataType, Frame>,
      internal_tags::deriv_inv_jacobian<DataType, Frame>,
      internal_tags::H<DataType>, internal_tags::kerr_schild_x<DataType, Frame>,
      internal_tags::a_cross_x<DataType, Frame>,
      internal_tags::kerr_schild_l<DataType, Frame>,
      internal_tags::l_lower<DataType, Frame>,
      internal_tags::l_upper<DataType, Frame>,
      internal_tags::deriv_r<DataType, Frame>,
      internal_tags::deriv_H<DataType, Frame>,
      internal_tags::kerr_schild_deriv_l<DataType, Frame>,
      internal_tags::deriv_l<DataType, Frame>,
      internal_tags::lapse_squared<DataType>, gr::Tags::Lapse<DataType>,
      internal_tags::deriv_lapse_multiplier<DataType>,
      internal_tags::shift_multiplier<DataType>,
      gr::Tags::Shift<DataType, 3, Frame>, DerivShift<DataType, Frame>,
      gr::Tags::SpatialMetric<DataType, 3, Frame>,
      DerivSpatialMetric<DataType, Frame>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>,
      gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>,
      gr::Tags::InverseSpatialMetric<DataType, 3, Frame>,
      gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame>,
      gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>>;

  // forward-declaration needed.
  template <typename DataType, typename Frame>
  class IntermediateVars;

  template <typename DataType, typename Frame = Frame::Inertial>
  using allowed_tags =
      tmpl::push_back<tags<DataType, Frame>,
                      typename internal_tags::inv_jacobian<DataType, Frame>>;

  template <typename DataType, typename Frame, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame>& x, double /*t*/,
      tmpl::list<Tags...> /*meta*/) const {
    static_assert(
        tmpl2::flat_all_v<
            tmpl::list_contains_v<allowed_tags<DataType, Frame>, Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");
    IntermediateVars<DataType, Frame> cache(get_size(*x.begin()));
    IntermediateComputer<DataType, Frame> computer(*this, x);
    return {cache.get_var(computer, Tags{})...};
  }

  template <typename DataType, typename Frame, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame>& x, double /*t*/,
      tmpl::list<Tags...> /*meta*/,
      gsl::not_null<IntermediateVars<DataType, Frame>*> cache) const {
    static_assert(
        tmpl2::flat_all_v<
            tmpl::list_contains_v<allowed_tags<DataType, Frame>, Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");
    if (cache->number_of_grid_points() != get_size(*x.begin())) {
      *cache = IntermediateVars<DataType, Frame>(get_size(*x.begin()));
    }
    IntermediateComputer<DataType, Frame> computer(*this, x);
    return {cache->get_var(computer, Tags{})...};
  }

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateComputer {
   public:
    using CachedBuffer = SphericalKerrSchild::CachedBuffer<DataType, Frame>;

    IntermediateComputer(const SphericalKerrSchild& solution,
                         const tnsr::I<DataType, 3, Frame>& x);

    // spin_a_and_squared(const SphericalKerrSchild& solution);

    void operator()(
        const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center,
        const gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_minus_center<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> r_squared,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r_squared<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> r,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> rho,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::rho<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_F,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::helper_matrix_F<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*>
            transformation_matrix_P,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::transformation_matrix_P<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> jacobian,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::jacobian<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_D,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::helper_matrix_D<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_C,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::helper_matrix_C<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::iJk<DataType, 3, Frame>*> deriv_jacobian,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_jacobian<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*>
            transformation_matrix_Q,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::transformation_matrix_Q<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_G1,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::helper_matrix_G1<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> a_dot_x,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_dot_x<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> s_number,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::s_number<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_G2,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::helper_matrix_G2<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::I<DataType, 3, Frame>*> G1_dot_x,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::G1_dot_x<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::i<DataType, 3, Frame>*> G2_dot_x,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::G2_dot_x<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> inv_jacobian,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::inv_jacobian<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_E1,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::helper_matrix_E1<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_E2,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::helper_matrix_E2<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::iJk<DataType, 3, Frame>*> deriv_inv_jacobian,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_inv_jacobian<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> H,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::H<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::I<DataType, 3, Frame>*> kerr_schild_x,
        const gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::kerr_schild_x<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::I<DataType, 3, Frame>*> a_cross_x,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_cross_x<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::I<DataType, 3, Frame>*> kerr_schild_l,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::kerr_schild_l<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::i<DataType, 4, Frame>*> l_lower,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::l_lower<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::I<DataType, 4, Frame>*> l_upper,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::l_upper<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::I<DataType, 3, Frame>*> deriv_r,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_r<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::I<DataType, 4, Frame>*> deriv_H,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_H<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ij<DataType, 4, Frame>*> kerr_schild_deriv_l,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::kerr_schild_deriv_l<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::ij<DataType, 4, Frame>*> deriv_l,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_l<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> lapse_squared,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::lapse_squared<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> lapse,
                    const gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::Lapse<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<Scalar<DataType>*> deriv_lapse_multiplier,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_lapse_multiplier<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> shift_multiplier,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::shift_multiplier<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
                    const gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::Shift<DataType, 3, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
        const gsl::not_null<CachedBuffer*> cache,
        DerivShift<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialMetric<DataType, 3, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::II<DataType, 3, Frame>*>
            inverse_spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::InverseSpatialMetric<DataType, 3, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*>
                        deriv_spatial_metric,
                    const gsl::not_null<CachedBuffer*> cache,
                    DerivSpatialMetric<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> extrinsic_curvature,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::ExtrinsicCurvature<DataType, 3, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*>
            christoffel_first_kind,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame> /*meta*/)
        const;

    void operator()(
        const gsl::not_null<tnsr::Ijj<DataType, 3, Frame>*>
            christoffel_second_kind,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame> /*meta*/)
        const;

   private:
    const SphericalKerrSchild& solution_;
    const tnsr::I<DataType, 3, Frame>& x_;
    // Here null_vector_0 is simply -1, but if you have a boosted solution,
    // then null_vector_0 can be something different, so we leave it coded
    // in instead of eliminating it.
    static constexpr double null_vector_0_ = -1.0;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateVars : public CachedBuffer<DataType, Frame> {
   public:
    using CachedBuffer = SphericalKerrSchild::CachedBuffer<DataType, Frame>;
    using CachedBuffer::CachedBuffer;
    using CachedBuffer::get_var;

    tnsr::i<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        DerivLapse<DataType, Frame> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/);

    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Shift<DataType, 3, Frame>> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/);

    tnsr::i<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::DerivDetSpatialMetric<DataType, 3, Frame> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/);

    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::TraceSpatialChristoffelSecondKind<DataType, 3,
                                                    Frame> /*meta*/);

   private:
    // Here null_vector_0 is simply -1, but if you have a boosted solution,
    // then null_vector_0 can be something different, so we leave it coded
    // in instead of eliminating it.
    static constexpr double null_vector_0_ = -1.0;
  };

 private:
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, volume_dim> dimensionless_spin_ =
      make_array<volume_dim>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, volume_dim> center_ =
      make_array<volume_dim>(std::numeric_limits<double>::signaling_NaN());
};

bool operator==(const SphericalKerrSchild& lhs, const SphericalKerrSchild& rhs);

bool operator!=(const SphericalKerrSchild& lhs, const SphericalKerrSchild& rhs);

}  // namespace Solutions
}  // namespace gr
