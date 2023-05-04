// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
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
 * \brief Kerr black hole in Kerr-Schild coordinates
 *
 * \details
 * The metric is \f$g_{\mu\nu} = \eta_{\mu\nu} + 2 H l_\mu l_\nu\f$,
 * where \f$\eta_{\mu\nu}\f$ is the Minkowski metric, \f$H\f$ is a scalar
 * function, and \f$l_\mu\f$ is the outgoing null vector.
 * \f$H\f$ and \f$l_\mu\f$ are known functions of the coordinates
 * and of the mass and spin vector.
 *
 * The following are input file options that can be specified:
 *  - Mass (default: 1.)
 *  - Center (default: {0,0,0})
 *  - Spin (default: {0,0,0})
 *
 * ## Kerr-Schild Coordinates
 *
 *
 * A Kerr-Schild coordinate system is defined by
 * \f{equation}{
 * g_{\mu\nu}  \equiv \eta_{\mu\nu} + 2 H l_\mu l_\nu,
 * \f}
 * where \f$H\f$ is a scalar function of the coordinates, \f$\eta_{\mu\nu}\f$
 * is the Minkowski metric, and \f$l^\mu\f$ is a null vector. Note that the
 * form of the metric along with the nullness of \f$l^\mu\f$ allows one to
 * raise and lower indices of \f$l^\mu\f$ using \f$\eta_{\mu\nu}\f$, and
 * that \f$l^t l^t = l_t l_t = l^i l_i\f$.
 * Note also that
 * \f{equation}{
 * g^{\mu\nu}  \equiv \eta^{\mu\nu} - 2 H l^\mu l^\nu,
 * \f}
 * and that \f$\sqrt{-g}=1\f$.
 * Also, \f$l_\mu\f$ is a geodesic with respect to both the physical metric
 * and the Minkowski metric:
 * \f{equation}{
 * l^\mu \partial_\mu l_\nu = l^\mu\nabla_\mu l_\nu = 0.
 * \f}
 *
 *
 * The corresponding 3+1 quantities are
 * \f{eqnarray}{
 * g_{i j}     &=& \delta_{i j} + 2 H l_i l_j,\\
 * g^{i j}  &=& \delta^{i j} - {2 H l^i l^j \over 1+2H l^t l^t},\\
 * {\rm det} g_{i j}&=& 1+2H l^t l^t,\\
 * \partial_k ({\rm det} g_{i j})&=& 2 l^t l^t \partial_k H,\\
 * \beta^i       &=& - {2 H l^t l^i \over 1+2H l^t l^t},\\
 * N        &=& \left(1+2 H l^t l^t\right)^{-1/2},\quad\hbox{(lapse)}\\
 * \alpha     &=& \left(1+2 H l^t l^t\right)^{-1},
 *                \quad\hbox{(densitized lapse)}\\
 * K_{i j}  &=& - \left(1+2 H l^t l^t\right)^{1/2}
 *       \left[l_i l_j \partial_{t} H + 2 H l_{(i} \partial_{t} l_{j)}\right]
 *                 \nonumber \\
 *                 &&-2\left(1+2 H l^t l^t\right)^{-1/2}
 *                \left[H l^t \partial_{(i}l_{j)} + H l_{(i}\partial_{j)}l^t
 *          + l^t l_{(i}\partial_{j)} H + 2H^2 l^t l_{(i} l^k\partial_{k}l_{j)}
 *          + H l^t l_i l_j l^k \partial_{k} H\right],\\
 * \partial_{k}g_{i j}&=& 2 l_i l_j\partial_{k} H +
 *    4 H l_{(i} \partial_{k}l_{j)},\\
 * \partial_{k}N   &=& -\left(1+2 H l^t l^t\right)^{-3/2}
 *                    \left(l^tl^t\partial_{k}H+2Hl^t\partial_{k}l^t\right),\\
 * \partial_{k}\beta^i  &=& - 2\left(1+2H l^t l^t\right)^{-1}
 *    \left(l^tl^i\partial_{k}H+Hl^t\partial_{k}l^i+Hl^i\partial_{k}l^t\right)
 *                   + 4 H l^t l^i \left(1+2H l^t l^t\right)^{-2}
 *                     \left(l^tl^t\partial_{k}H+2Hl^t\partial_{k}l^t\right),\\
 * \Gamma^k{}_{i j}&=& -\delta^{k m}\left(l_i l_j \partial_{m}H
 *                                    + 2l_{(i} \partial_{m}l_{j)} \right)
 *                   + 2 H l_{(i}\partial_{j)} l^k
 *          \nonumber \\
 *                  &&+\left(1+2 H l^t l^t\right)^{-1}
 *                     \left[2 l^k l_{(i}\partial_{j)} H
 *                          +2 H l_i l_j l^k l^m \partial_{m}H
 *                          +2 H l^k \partial_{(i}l_{j)}
 *                          +4 H^2 l^k l_{(i} (l^m \partial_{m}l_{j)}
 *                                      -\partial_{j)} l^t)
 *                     \right].
 * \f}
 * Note that \f$l^i\f$ is **not** equal to \f$g^{i j} l_j\f$; it is equal
 * to \f${}^{(4)}g^{i \mu} l_\mu\f$.
 *
 * ## Kerr Spacetime
 *
 * ### Spin in the z direction
 *
 * Assume Cartesian coordinates \f$(t,x,y,z)\f$. Then for stationary Kerr
 * spacetime with mass \f$M\f$ and angular momentum \f$a M\f$
 * in the \f$z\f$ direction,
 * \f{eqnarray}{
 * H     &=& {M r^3 \over r^4 + a^2 z^2},\\
 * l_\mu &=&
 * \left(1,{rx+ay\over r^2+a^2},{ry-ax\over r^2+a^2},{z\over r}\right),
 * \f}
 * where \f$r\f$ is defined by
 * \f{equation}{
 * \label{eq:rdefinition1}
 * {x^2+y^2\over a^2+r^2} + {z^2\over r^2} = 1,
 * \f}
 * or equivalently,
 * \f{equation}{
 * r^2 = {1\over 2}(x^2 + y^2 + z^2 - a^2)
 *      + \left({1\over 4}(x^2 + y^2 + z^2 - a^2)^2 + a^2 z^2\right)^{1/2}.
 * \f}
 *
 * Possibly useful formula:
 * \f{equation}{
 * \partial_{i} r = {x_i + z \delta_{i z} \displaystyle {a^2\over r^2} \over
 *   2 r\left(1 - \displaystyle {x^2 + y^2 + z^2 - a^2\over 2 r^2}\right)}.
 * \f}
 *
 * ### Spin in an arbitrary direction
 *
 * For arbitrary spin direction, let \f$\vec{x}\equiv (x,y,z)\f$ and
 * \f$\vec{a}\f$ be a flat-space three-vector with magnitude-squared
 * (\f$\delta_{ij}\f$ norm) equal to \f$a^2\f$.
 * Then the Kerr-Schild quantities for Kerr spacetime are:
 * \f{eqnarray}{
 * H       &=& {M r^3 \over r^4 + (\vec{a}\cdot\vec{x})^2},\\
 * \vec{l} &=& {r\vec{x}-\vec{a}\times\vec{x}+(\vec{a}\cdot\vec{x})\vec{a}/r
 *              \over r^2+a^2 },\\
 * l_t     &=& 1,\\
 * \label{eq:rdefinition2}
 * r^2     &=& {1\over 2}(\vec{x}\cdot\vec{x}-a^2)
 *      + \left({1\over 4}(\vec{x}\cdot\vec{x}-a^2)^2
 *              + (\vec{a}\cdot\vec{x})^2\right)^{1/2},
 * \f}
 * where \f$\vec{l}\equiv (l_x,l_y,l_z)\f$, and
 * all dot and cross products are evaluated as flat-space 3-vector operations.
 *
 * Possibly useful formulae:
 * \f{equation}{
 * \partial_{i} r = {x_i + (\vec{a}\cdot\vec{x})a_i/r^2 \over
 *   2 r\left(1 - \displaystyle {\vec{x}\cdot\vec{x}-a^2\over 2 r^2}\right)},
 * \f}
 * \f{equation}{
 * {\partial_{i} H \over H} =
 *  {3\partial_{i}r\over r} - {4 r^3 \partial_{i}r +
 *                     2(\vec{a}\cdot\vec{x})\vec{a}
 *                     \over r^4 + (\vec{a}\cdot\vec{x})^2},
 * \f}
 * \f{equation}{
 * (r^2+a^2)\partial_{j} l_i =
 *                 (x_i-2 r l_i-(\vec{a}\cdot\vec{x})a_i/r^2)\partial_{j}r
 *                 + r\delta_{ij} + a_i a_j/r + \epsilon^{ijk} a_k.
 * \f}
 *
 * ## Cartesian and Spherical Coordinates for Kerr
 *
 * The Kerr-Schild coordinates are defined in terms of the Cartesian
 * coordinates \f$(x,y,z)\f$.  If one wishes to express Kerr-Schild
 * coordinates in terms of the spherical polar coordinates
 * \f$(\tilde{r},\theta,\phi)\f$ then one can make the obvious and
 * usual transformation
 * \f{equation}{
 * \label{eq:sphertocartsimple}
 * x=\tilde{r}\sin\theta\cos\phi,\quad
 * y=\tilde{r}\sin\theta\sin\phi,\quad
 * z=\tilde{r}\cos\theta.
 * \f}
 *
 * This is simple, and has the advantage that in this coordinate system
 * for \f$M\to0\f$, Kerr spacetime becomes Minkowski space in spherical
 * coordinates \f$(\tilde{r},\theta,\phi)\f$. However, the disadvantage is
 * that the horizon of a Kerr hole is **not** located at constant
 * \f$\tilde{r}\f$, but is located instead at constant \f$r\f$,
 * where \f$r\f$ is the radial
 * Boyer-Lindquist coordinate defined in (\f$\ref{eq:rdefinition2}\f$).
 *
 * For spin in the \f$z\f$ direction, one could use the transformation
 * \f{equation}{
 * x=\sqrt{r^2+a^2}\sin\theta\cos\phi,\quad
 * y=\sqrt{r^2+a^2}\sin\theta\sin\phi,\quad
 * z=r\cos\theta.
 * \f}
 * In this case, for \f$M\to0\f$, Kerr spacetime becomes Minkowski space in
 * spheroidal coordinates, but now the horizon is on a constant-coordinate
 * surface.
 *
 * Right now we use (\f$\ref{eq:sphertocartsimple}\f$), but we may
 * wish to use the other transformation in the future.
 */
class KerrSchild : public AnalyticSolution<3_st>,
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
      "Black hole in Kerr-Schild coordinates"};

  KerrSchild(double mass, const std::array<double, 3>& dimensionless_spin,
             const std::array<double, 3>& center,
             const Options::Context& context = {});

  /// \cond
  explicit KerrSchild(CkMigrateMessage* /*msg*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(KerrSchild);
  /// \endcond

  template <typename DataType, typename Frame = Frame::Inertial>
  using tags = tmpl::flatten<tmpl::list<
      AnalyticSolution<3_st>::tags<DataType, Frame>,
      gr::Tags::DerivDetSpatialMetric<3, Frame, DataType>,
      gr::Tags::TraceExtrinsicCurvature<DataType>,
      gr::Tags::SpatialChristoffelFirstKind<3, Frame, DataType>,
      gr::Tags::SpatialChristoffelSecondKind<3, Frame, DataType>,
      gr::Tags::TraceSpatialChristoffelSecondKind<3, Frame, DataType>>>;

  KerrSchild() = default;
  KerrSchild(const KerrSchild& /*rhs*/) = default;
  KerrSchild& operator=(const KerrSchild& /*rhs*/) = default;
  KerrSchild(KerrSchild&& /*rhs*/) = default;
  KerrSchild& operator=(KerrSchild&& /*rhs*/) = default;
  ~KerrSchild() = default;

  std::unique_ptr<evolution::initial_data::InitialData> get_clone()
      const override {
    return std::make_unique<KerrSchild>(*this);
  }

  template <typename DataType, typename Frame, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame>& x, double /*t*/,
      tmpl::list<Tags...> /*meta*/) const {
    static_assert(
        tmpl2::flat_all_v<
            tmpl::list_contains_v<tags<DataType, Frame>, Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");
    IntermediateVars<DataType, Frame> cache(get_size(*x.begin()));
    IntermediateComputer<DataType, Frame> computer(*this, x);
    return {cache.get_var(computer, Tags{})...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  double mass() const { return mass_; }
  const std::array<double, volume_dim>& center() const { return center_; }
  const std::array<double, volume_dim>& dimensionless_spin() const {
    return dimensionless_spin_;
  }
  bool zero_spin() const { return zero_spin_; }

  struct internal_tags {
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_minus_center = ::Tags::TempI<0, 3, Frame, DataType>;
    template <typename DataType>
    using a_dot_x = ::Tags::TempScalar<1, DataType>;
    template <typename DataType>
    using a_dot_x_squared = ::Tags::TempScalar<2, DataType>;
    template <typename DataType>
    using half_xsq_minus_asq = ::Tags::TempScalar<3, DataType>;
    template <typename DataType>
    using r_squared = ::Tags::TempScalar<4, DataType>;
    template <typename DataType>
    using r = ::Tags::TempScalar<5, DataType>;
    template <typename DataType>
    using a_dot_x_over_rsquared = ::Tags::TempScalar<6, DataType>;
    template <typename DataType>
    using deriv_log_r_denom = ::Tags::TempScalar<7, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_log_r = ::Tags::Tempi<8, 3, Frame, DataType>;
    template <typename DataType>
    using H_denom = ::Tags::TempScalar<9, DataType>;
    template <typename DataType>
    using H = ::Tags::TempScalar<10, DataType>;
    template <typename DataType>
    using deriv_H_temp1 = ::Tags::TempScalar<11, DataType>;
    template <typename DataType>
    using deriv_H_temp2 = ::Tags::TempScalar<12, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_H = ::Tags::Tempi<13, 3, Frame, DataType>;
    template <typename DataType>
    using denom = ::Tags::TempScalar<14, DataType>;
    template <typename DataType>
    using a_dot_x_over_r = ::Tags::TempScalar<15, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using null_form = ::Tags::Tempi<16, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_null_form = ::Tags::Tempij<17, 3, Frame, DataType>;
    template <typename DataType>
    using lapse_squared = ::Tags::TempScalar<18, DataType>;
    template <typename DataType>
    using deriv_lapse_multiplier = ::Tags::TempScalar<19, DataType>;
    template <typename DataType>
    using shift_multiplier = ::Tags::TempScalar<20, DataType>;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  using CachedBuffer = CachedTempBuffer<
      internal_tags::x_minus_center<DataType, Frame>,
      internal_tags::a_dot_x<DataType>,
      internal_tags::a_dot_x_squared<DataType>,
      internal_tags::half_xsq_minus_asq<DataType>,
      internal_tags::r_squared<DataType>, internal_tags::r<DataType>,
      internal_tags::a_dot_x_over_rsquared<DataType>,
      internal_tags::deriv_log_r_denom<DataType>,
      internal_tags::deriv_log_r<DataType, Frame>,
      internal_tags::H_denom<DataType>, internal_tags::H<DataType>,
      internal_tags::deriv_H_temp1<DataType>,
      internal_tags::deriv_H_temp2<DataType>,
      internal_tags::deriv_H<DataType, Frame>, internal_tags::denom<DataType>,
      internal_tags::a_dot_x_over_r<DataType>,
      internal_tags::null_form<DataType, Frame>,
      internal_tags::deriv_null_form<DataType, Frame>,
      internal_tags::lapse_squared<DataType>, gr::Tags::Lapse<DataType>,
      internal_tags::deriv_lapse_multiplier<DataType>,
      internal_tags::shift_multiplier<DataType>,
      gr::Tags::Shift<3, Frame, DataType>, DerivShift<DataType, Frame>,
      gr::Tags::SpatialMetric<3, Frame, DataType>,
      gr::Tags::InverseSpatialMetric<3, Frame, DataType>,
      DerivSpatialMetric<DataType, Frame>,
      ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>>,
      gr::Tags::ExtrinsicCurvature<3, Frame, DataType>,
      gr::Tags::SpatialChristoffelFirstKind<3, Frame, DataType>,
      gr::Tags::SpatialChristoffelSecondKind<3, Frame, DataType>>;

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateComputer {
   public:
    using CachedBuffer = KerrSchild::CachedBuffer<DataType, Frame>;

    IntermediateComputer(const KerrSchild& solution,
                         const tnsr::I<DataType, 3, Frame>& x);

    const KerrSchild& solution() const { return solution_; }

    void operator()(
        const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center,
        const gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_minus_center<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> a_dot_x,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_dot_x<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> a_dot_x_squared,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_dot_x_squared<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> half_xsq_minus_asq,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::half_xsq_minus_asq<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> r_squared,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r_squared<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> r,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<Scalar<DataType>*> a_dot_x_over_rsquared,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::a_dot_x_over_rsquared<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> deriv_log_r_denom,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_log_r_denom<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_log_r,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_log_r<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> H_denom,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::H_denom<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> H,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::H<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> deriv_H_temp1,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_H_temp1<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> deriv_H_temp2,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_H_temp2<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_H,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_H<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> denom,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::denom<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> a_dot_x_over_r,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_dot_x_over_r<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::i<DataType, 3, Frame>*> null_form,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::null_form<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ij<DataType, 3, Frame>*> deriv_null_form,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_null_form<DataType, Frame> /*meta*/) const;

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
                    gr::Tags::Shift<3, Frame, DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
        const gsl::not_null<CachedBuffer*> cache,
        DerivShift<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialMetric<3, Frame, DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::II<DataType, 3, Frame>*> spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::InverseSpatialMetric<3, Frame, DataType> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*>
                        deriv_spatial_metric,
                    const gsl::not_null<CachedBuffer*> cache,
                    DerivSpatialMetric<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> extrinsic_curvature,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::ExtrinsicCurvature<3, Frame, DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*>
            spatial_christoffel_first_kind,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialChristoffelFirstKind<3, Frame, DataType> /*meta*/)
        const;

    void operator()(
        const gsl::not_null<tnsr::Ijj<DataType, 3, Frame>*>
            spatial_christoffel_second_kind,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialChristoffelSecondKind<3, Frame, DataType> /*meta*/)
        const;

   private:
    const KerrSchild& solution_;
    const tnsr::I<DataType, 3, Frame>& x_;
    // Here null_vector_0 is simply -1, but if you have a boosted solution,
    // then null_vector_0 can be something different, so we leave it coded
    // in instead of eliminating it.
    static constexpr double null_vector_0_ = -1.0;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateVars : public CachedBuffer<DataType, Frame> {
   public:
    using CachedBuffer = KerrSchild::CachedBuffer<DataType, Frame>;
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
        ::Tags::dt<gr::Tags::Shift<3, Frame, DataType>> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/);

    tnsr::i<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::DerivDetSpatialMetric<3, Frame, DataType> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/);

    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::TraceSpatialChristoffelSecondKind<3, Frame,
                                                    DataType> /*meta*/);

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
  bool zero_spin_{};
};

SPECTRE_ALWAYS_INLINE bool operator==(const KerrSchild& lhs,
                                      const KerrSchild& rhs) {
  return lhs.mass() == rhs.mass() and
         lhs.dimensionless_spin() == rhs.dimensionless_spin() and
         lhs.center() == rhs.center();
}

SPECTRE_ALWAYS_INLINE bool operator!=(const KerrSchild& lhs,
                                      const KerrSchild& rhs) {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace gr
