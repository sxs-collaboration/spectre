// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Literals.hpp"
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
 * \brief Schwarzschild black hole in Cartesian coordinates with harmonic gauge
 *
 * \details
 * This solution represents a Schwarzschild black hole in coordinates that are
 * harmonic in both space and time, as well as horizon-penetrating. Therefore,
 * this solution fulfills the harmonic coordinate conditions Eq. (4.42), (4.44),
 * and (4.45) in \cite BaumgarteShapiro :
 *
 * \f{align}
 *     {}^{(4)}\Gamma^a &= 0
 * \f}
 *
 * \f{align}
 *     (\partial_t - \beta^j \partial_j)\alpha &= -\alpha^2 K
 * \f}
 *
 * \f{align}
 *     (\partial_t - \beta^j \partial_j)\beta^i &=
 *         -\alpha^2 (\gamma^{ij} \partial_j \ln \alpha -
 *                    \gamma^{jk} \Gamma^i_{jk})
 * \f}
 *
 * (Note that Eq. 4.45 in \cite BaumgarteShapiro is missing a minus sign in
 * front of the \f$\Gamma\f$-contraction term)
 *
 * We implement Eqs. (45)--(50) in \cite Cook1997qc , which represent the
 * zero-spin limit of the time-harmonic and horizon-penetrating slices of Kerr
 * spacetime presented in the paper. We add the radial transformation
 * \f$r \to r + M\f$ to make the spatial coordinates harmonic as well (see
 * Eq. (43) in \cite Cook1997qc ), so the coordinates remain harmonic under
 * boosts.
 *
 * Consider a Schwarzschild black hole of mass \f$M\f$ and center \f$C^i\f$. The
 * spacetime will be specified using Cartesian coordinates \f$x^i\f$ that are
 * spatially and temporally harmonic. A radius centered on the black hole is
 *
 * \f{align}
 *     r &= \sqrt{\delta_{ij} \left(x^i - C^i\right)\left(x^j - C^j\right)}
 * \f}
 *
 * For computing the spatial metric, we define the following quantities:
 *
 * \f{align}
 *     \gamma_{rr} &= 1 + \frac{2M}{M+r} + \left(\frac{2M}{M+r}\right)^2
 *         + \left(\frac{2M}{M+r}\right)^3,\\
 *     \partial_r \gamma_{rr} &= -\frac{1}{2M}\left(\frac{2M}{M+r}\right)^2
 *         -\frac{1}{M}\left(\frac{2M}{M+r}\right)^3
 *         -\frac{3}{2M}\left(\frac{2M}{M+r}\right)^4,\\
 *     \frac{X^i}{r} &= \frac{x^i - C^i}{r},\\
 *     X_j &= X^i \delta_{ij}
 * \f}
 *
 * From these quantities, the spatial metric and its time derivative are
 * computed as
 *
 * \f{align}
 *     \gamma_{ij} &=
 *         \left(\gamma_{rr} - \left(1+\frac{M}{r}\right)^2\right)
 *             \frac{X_i}{r} \frac{X_j}{r} +
 *         \delta_{ij} \left(1+\frac{M}{r}\right)^2,\\
 *     \partial_t \gamma_{ij} &= 0
 * \f}
 *
 * The spatial derivative is given in terms of the following quantities:
 *
 * \f{align}
 *     f_0 &= \left(1+\frac{M}{r}\right)^2\\
 *     \partial_r f_0 &=
 *         2 \left(1+\frac{M}{r}\right)\left(-\frac{M}{r^2}\right),\\
 *     f_1 &= \frac{1}{r} \left(\gamma_{rr} - f_0\right),\\
 *     f_2 &= \partial_r \gamma_{rr} - \partial_r f_0 - 2 f_1
 * \f}
 *
 * In terms of these, the spatial metric's spatial derivative is
 *
 * \f{align}
 *     \partial_k \gamma_{ij} &=
 *         f_2 \frac{X_i}{r} \frac{X_j}{r} \frac{X_k}{r} +
 *         f_1 \frac{X_j}{r} \delta_{ik} + f_1 \frac{X_i}{r} \delta_{jk} +
 *         \partial_r f_0 \frac{X_k}{r} \delta_{ij}
 * \f}
 *
 * The lapse and its derivatives are
 *
 * \f{align}
 *     \alpha &= \gamma_{rr}^{-1/2},\\
 *     \partial_t \alpha &= 0,\\
 *     \partial_i \alpha &=
 *         -\frac{1}{2} \gamma_{rr}^{-3/2} \partial_r \gamma_{rr} \frac{X_i}{r}
 * \f}
 *
 * The shift and its time derivative are
 *
 * \f{align}
 *     \beta^i &= \left(\frac{2M}{M+r}\right)^2 \frac{X^i}{r}
 *         \frac{1}{\gamma_{rr}},\\
 *     \partial_t \beta^i &= 0
 * \f}
 *
 * The spatial derivative of the shift is computed in terms of the following
 * quantities:
 *
 * \f{align}
 *     f_3 &= \frac{1}{r} \frac{1}{\gamma_{rr}} \left(\frac{2M}{M+r}\right)^2,\\
 *     f_4 &=
 *         -f_3 -
 *         \frac{1}{M} \frac{1}{\gamma_{rr}}
 *             \left(\frac{2M}{M+r}\right)^3 -
 *         \partial_r \gamma_{rr} \left(\frac{2 M}{M+r}
 *             \frac{1}{\gamma_{rr}}\right)^2
 * \f}
 *
 * In terms of these, the shift's spatial derivative is
 *
 * \f{align}
 *     \partial_k \beta^i &= f_4 \frac{X^i}{r} \frac{X_k}{r} + \delta_k^i f_3
 * \f}
 */
class HarmonicSchwarzschild : public AnalyticSolution<3_st>,
                              public MarkAsAnalyticSolution {
 public:
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"Mass of the black hole"};
    static type lower_bound() { return 0.; }
  };
  struct Center {
    using type = std::array<double, volume_dim>;
    static constexpr Options::String help = {
        "The [x,y,z] center of the black hole"};
  };
  using options = tmpl::list<Mass, Center>;
  static constexpr Options::String help{
      "Schwarzschild black hole in Cartesian coordinates with harmonic gauge"};

  HarmonicSchwarzschild(double mass,
                        const std::array<double, volume_dim>& center,
                        const Options::Context& context = {});

  HarmonicSchwarzschild() = default;
  HarmonicSchwarzschild(const HarmonicSchwarzschild& /*rhs*/) = default;
  HarmonicSchwarzschild& operator=(const HarmonicSchwarzschild& /*rhs*/) =
      default;
  HarmonicSchwarzschild(HarmonicSchwarzschild&& /*rhs*/) = default;
  HarmonicSchwarzschild& operator=(HarmonicSchwarzschild&& /*rhs*/) = default;
  ~HarmonicSchwarzschild() = default;

  explicit HarmonicSchwarzschild(CkMigrateMessage* /*msg*/);

  /*!
   * \brief Computes and returns spacetime quantities for a Schwarzschild black
   * hole with harmonic coordinates at a specific Cartesian position
   *
   * \param x Cartesian coordinates of the position at which to compute
   * spacetime quantities
   */
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
  void pup(PUP::er& p);

  /*!
   * \brief Return the mass of the black hole
   */
  SPECTRE_ALWAYS_INLINE double mass() const { return mass_; }
  /*!
   * \brief Return the center of the black hole
   */
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>& center() const {
    return center_;
  }

  /*!
   * \brief Tags defined for intermediates specific to the harmonic
   * Schwarzschild solution
   */
  struct internal_tags {
    /*!
     * \brief Tag for the position of a point relative to the center of the
     * black hole
     *
     * \details Defined as \f$X^i = \left(x^i - C^i\right)\f$, where \f$C^i\f$
     * is the Cartesian coordinates of the center of the black hole
     * and \f$x^i\f$ is the Cartesian coordinates of the point where we're
     * wanting to compute spacetime quantities.
     */
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_minus_center = ::Tags::TempI<0, 3, Frame, DataType>;
    /*!
     * \brief Tag for the radius corresponding to the position of a point
     * relative to the center of the black hole
     *
     * \details Defined as \f$r = \sqrt{\delta_{ij} X^i X^j}\f$, where \f$X^i\f$
     * is defined by `internal_tags::x_minus_center`.
     */
    template <typename DataType>
    using r = ::Tags::TempScalar<1, DataType>;
    /*!
     * \brief Tag for one over the radius corresponding to the position of a
     * point relative to the center of the black hole
     *
     * \details The quantity \f$r\f$ is the radius defined by
     * `internal_tags::r`.
     */
    template <typename DataType>
    using one_over_r = ::Tags::TempScalar<2, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\frac{X^i}{r}\f$
     *
     * \details The quantity \f$X^i\f$ is defined by
     * `internal_tags::x_minus_center` and \f$r\f$ is the radius defined by
     * `internal_tags::r`.
     */
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_over_r = ::Tags::TempI<3, 3, Frame, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\frac{M}{r}\f$
     *
     * \details The quantity \f$M\f$ is the mass of the black hole and \f$r\f$
     * is the radius defined by `internal_tags::r`.
     */
    template <typename DataType>
    using m_over_r = ::Tags::TempScalar<4, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\sqrt{f_0} = 1 + \frac{M}{r}\f$
     *
     * \details The quantity \f$M\f$ is the mass of the black hole, \f$r\f$ is
     * the radius defined by `internal_tags::r`, and \f$f_0\f$ is defined by
     * `internal_tags::f_0`.
     */
    template <typename DataType>
    using sqrt_f_0 = ::Tags::TempScalar<5, DataType>;
    /*!
     * \brief Tag for the intermediate
     * \f$f_0 = \left(1 + \frac{M}{r}\right)^2\f$
     *
     * \details The quantity \f$M\f$ is the mass of the black hole and \f$r\f$
     * is the radius defined by `internal_tags::r`.
     */
    template <typename DataType>
    using f_0 = ::Tags::TempScalar<6, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\frac{2M}{M+r}\f$
     *
     * \details The quantity \f$M\f$ is the mass of the black hole and \f$r\f$
     * is the radius defined by `internal_tags::r`.
     */
    template <typename DataType>
    using two_m_over_m_plus_r = ::Tags::TempScalar<7, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\left(\frac{2M}{M+r}\right)^2\f$
     *
     * \details The quantity \f$M\f$ is the mass of the black hole and \f$r\f$
     * is the radius defined by `internal_tags::r`.
     */
    template <typename DataType>
    using two_m_over_m_plus_r_squared = ::Tags::TempScalar<8, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\left(\frac{2M}{M+r}\right)^3\f$
     *
     * \details The quantity \f$M\f$ is the mass of the black hole and \f$r\f$
     * is the radius defined by `internal_tags::r`.
     */
    template <typename DataType>
    using two_m_over_m_plus_r_cubed = ::Tags::TempScalar<9, DataType>;
    /*!
     * \brief Tag for the \f$\gamma_{rr}\f$ component of the spatial metric
     *
     * \details Defined as
     *
     * \f{align}
     *     \gamma_{rr} &= 1 + \frac{2M}{M+r} + \left(\frac{2M}{M+r}\right)^2
     *         + \left(\frac{2M}{M+r}\right)^3
     * \f}
     *
     * where \f$M\f$ is the mass of the black hole and \f$r\f$ is the radius
     * defined by `internal_tags::r`.
     */
    template <typename DataType>
    using spatial_metric_rr = ::Tags::TempScalar<10, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\frac{1}{\gamma_{rr}}\f$
     *
     * \details The quantity \f$\gamma_{rr}\f$ is defined by
     * `internal_tags::spatial_metric_rr`.
     */
    template <typename DataType>
    using one_over_spatial_metric_rr = ::Tags::TempScalar<11, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\gamma_{rr} - f_0\f$
     *
     * \details The quantity \f$\gamma_{rr}\f$ is defined by
     * `internal_tags::spatial_metric_rr` and \f$f_0\f$ is defined by
     * `internal_tags::f_0`.
     */
    template <typename DataType>
    using spatial_metric_rr_minus_f_0 = ::Tags::TempScalar<12, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\partial_r \gamma_{rr}\f$
     *
     * \details Defined as
     *
     * \f{align}
     *     \partial_r \gamma_{rr} &= -\frac{1}{2M}\left(\frac{2M}{M+r}\right)^2
     *         -\frac{1}{M}\left(\frac{2M}{M+r}\right)^3
     *         -\frac{3}{2M}\left(\frac{2M}{M+r}\right)^4
     * \f}
     *
     * where \f$M\f$ is the mass of the black hole and \f$r\f$ is the radius
     * defined by `internal_tags::r`.
     */
    template <typename DataType>
    using d_spatial_metric_rr = ::Tags::TempScalar<13, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\partial_r f_0\f$
     *
     * \details Defined as
     *
     * \f{align}
     *     \partial_r f_0 &=
     *         2 \left(1+\frac{M}{r}\right)\left(-\frac{M}{r^2}\right)
     * \f}
     *
     * where \f$M\f$ is the mass of the black hole and \f$r\f$ is the radius
     * defined by `internal_tags::r`.
     */
    template <typename DataType>
    using d_f_0 = ::Tags::TempScalar<14, DataType>;
    /*!
     * \brief Tag for the intermediate \f$\partial_r f_0 \frac{X_i}{r}\f$
     *
     * \details The quantity \f$r\f$ is the radius defined by
     * `internal_tags::r`, \f$\partial_r f_0\f$ is defined by
     * `internal_tags::d_f_0`, and \f$X_j = X^i \delta_{ij}\f$ where \f$X^i\f$
     * is defined by `internal_tags::x_minus_center`.
     */
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using d_f_0_times_x_over_r = ::Tags::Tempi<15, 3, Frame, DataType>;
    /*!
     * \brief Tag for the intermediate
     * \f$f_1 = \frac{1}{r} \left(\gamma_{rr} - f_0\right)\f$
     *
     * \details The quantity \f$r\f$ is the radius defined by
     * `internal_tags::r`, \f$\gamma_{rr}\f$ is defined by
     * `internal_tags::spatial_metric_rr`, and \f$f_0\f$ is defined by
     * `internal_tags::f_0`.
     */
    template <typename DataType>
    using f_1 = ::Tags::TempScalar<16, DataType>;
    /*!
     * \brief Tag for the intermediate \f$f_1 \frac{X_i}{r}\f$
     *
     * \details The quantity \f$r\f$ is the radius defined by
     * `internal_tags::r`, \f$f_1\f$ is defined by `internal_tags::f_1`, and
     * \f$X_j = X^i \delta_{ij}\f$ where \f$X^i\f$ is defined by
     * `internal_tags::x_minus_center`.
     */
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using f_1_times_x_over_r = ::Tags::Tempi<17, 3, Frame, DataType>;
    /*!
     * \brief Tag for the intermediate
     * \f$f_2 = \partial_r \gamma_{rr} - \partial_r f_0 - 2 f_1\f$
     *
     * \details The quantity \f$\partial_r \gamma_{rr}\f$ is defined by
     * `internal_tags::d_spatial_metric_rr`, \f$\partial_r f_0\f$ is defined by
     * `internal_tags::d_f_0`, and \f$f_1\f$ is defined by `internal_tags::f_1`.
     */
    template <typename DataType>
    using f_2 = ::Tags::TempScalar<18, DataType>;
    /*!
     * \brief Tag for the intermediate
     * \f$f_2 \frac{X_i}{r} \frac{X_j}{r} \frac{X_k}{r}\f$
     *
     * \details The quantity \f$r\f$ is the radius defined by
     * `internal_tags::r`, \f$f_2\f$ is defined by `internal_tags::f_2`, and
     * \f$X_j = X^i \delta_{ij}\f$ where \f$X^i\f$ is defined by
     * `internal_tags::x_minus_center`.
     */
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using f_2_times_xxx_over_r_cubed = ::Tags::Tempiii<19, 3, Frame, DataType>;
    /*!
     * \brief Tag for the intermediate
     * \f$f_3 = \frac{1}{r}\frac{1}{\gamma_{rr}}\left(\frac{2M}{M+r}\right)^2\f$
     *
     * \details The quantity \f$M\f$ is the mass of the black hole, \f$r\f$ is
     * the radius defined by `internal_tags::r`, and \f$\gamma_{rr}\f$ is
     * defined by `internal_tags::spatial_metric_rr`.
     */
    template <typename DataType>
    using f_3 = ::Tags::TempScalar<20, DataType>;
    /*!
     * \brief Tag for the intermediate
     *
     * \f{align}
     *     f_4 &=
     *         -f_3 -
     *         \frac{1}{M} \frac{1}{\gamma_{rr}}
     *             \left(\frac{2M}{M+r}\right)^3 -
     *         \partial_r \gamma_{rr} \left(\frac{2 M}{M+r}
     *             \frac{1}{\gamma_{rr}}\right)^2
     * \f}
     *
     * \details The quantity \f$M\f$ is the mass of the black hole, \f$r\f$ is
     * the radius defined by `internal_tags::r`, \f$f_3\f$ is defined by
     * `internal_tags::f_3`, \f$\gamma_{rr}\f$ is defined by
     * `internal_tags::spatial_metric_rr`, and its derivative
     * \f$\partial_r \gamma_{rr}\f$ is defined by
     * `internal_tags::d_spatial_metric_rr`.
     */
    template <typename DataType>
    using f_4 = ::Tags::TempScalar<21, DataType>;
    /*!
     * \brief Tag for one over the determinant of the spatial metric
     */
    template <typename DataType>
    using one_over_det_spatial_metric = ::Tags::TempScalar<22, DataType>;
    /*!
     * \brief Tag for the intermediate
     * \f$-\frac{1}{2} \gamma_{rr}^{-3/2} \partial_r \gamma_{rr}\f$
     *
     * \details The lapse is defined as \f$\alpha = \gamma_{rr}^{-1/2}\f$,
     * \f$\gamma_{rr}\f$ is defined by `internal_tags::spatial_metric_rr` and
     * its derivative \f$\partial_r \gamma_{rr}\f$ is defined by
     * `internal_tags::d_spatial_metric_rr`.
     */
    template <typename DataType>
    using neg_half_lapse_cubed_times_d_spatial_metric_rr =
        ::Tags::TempScalar<23, DataType>;
  };

  /*!
   * \brief Buffer for caching computed intermediates and quantities that we do
   * not want to recompute across the solution's implementation
   *
   * \details See `internal_tags` documentation for details on what quantities
   * the internal tags represent
   */
  template <typename DataType, typename Frame = ::Frame::Inertial>
  using CachedBuffer = CachedTempBuffer<
      internal_tags::x_minus_center<DataType, Frame>,
      internal_tags::r<DataType>, internal_tags::one_over_r<DataType>,
      internal_tags::x_over_r<DataType, Frame>,
      internal_tags::m_over_r<DataType>, internal_tags::sqrt_f_0<DataType>,
      internal_tags::f_0<DataType>,
      internal_tags::two_m_over_m_plus_r<DataType>,
      internal_tags::two_m_over_m_plus_r_squared<DataType>,
      internal_tags::two_m_over_m_plus_r_cubed<DataType>,
      internal_tags::spatial_metric_rr<DataType>,
      internal_tags::one_over_spatial_metric_rr<DataType>,
      internal_tags::spatial_metric_rr_minus_f_0<DataType>,
      internal_tags::d_spatial_metric_rr<DataType>,
      internal_tags::d_f_0<DataType>,
      internal_tags::d_f_0_times_x_over_r<DataType, Frame>,
      internal_tags::f_1<DataType>,
      internal_tags::f_1_times_x_over_r<DataType, Frame>,
      internal_tags::f_2<DataType>,
      internal_tags::f_2_times_xxx_over_r_cubed<DataType, Frame>,
      internal_tags::f_3<DataType>, internal_tags::f_4<DataType>,
      gr::Tags::Lapse<DataType>,
      internal_tags::neg_half_lapse_cubed_times_d_spatial_metric_rr<DataType>,
      gr::Tags::Shift<DataType, 3, Frame>, DerivShift<DataType, Frame>,
      gr::Tags::SpatialMetric<DataType, 3, Frame>,
      DerivSpatialMetric<DataType, Frame>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>,
      gr::Tags::DetSpatialMetric<DataType>,
      internal_tags::one_over_det_spatial_metric<DataType>>;

  /*!
   * \brief Computes the intermediates and quantities that we do not want to
   * recompute across the solution's implementation
   */
  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateComputer {
   public:
    using CachedBuffer = HarmonicSchwarzschild::CachedBuffer<DataType, Frame>;

    /*!
     * \brief Constructs a computer for spacetime quantities of a given
     * `gr::Solutions::HarmonicSchwarzschild` solution at at a specific
     * Cartesian position
     *
     * \param solution the given `gr::Solutions::HarmonicSchwarzschild` solution
     * \param x Cartesian coordinates of the position at which to compute
     * spacetime quantities
     */
    IntermediateComputer(const HarmonicSchwarzschild& solution,
                         const tnsr::I<DataType, 3, Frame>& x);

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::x_minus_center`
     */
    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center,
        gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_minus_center<DataType, Frame> /*meta*/) const;

    /*!
     * \brief Computes the radius defined by `internal_tags::r`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::one_over_r`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> one_over_r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::one_over_r<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::x_over_r`
     */
    void operator()(gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_over_r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::x_over_r<DataType, Frame> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::m_over_r`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> m_over_r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::m_over_r<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::sqrt_f_0`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> sqrt_f_0,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::sqrt_f_0<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::f_0`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> f_0,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_0<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::two_m_over_m_plus_r`
     */
    void operator()(
        gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::two_m_over_m_plus_r<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::two_m_over_m_plus_r_squared`
     */
    void operator()(
        gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r_squared,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::two_m_over_m_plus_r_squared<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::two_m_over_m_plus_r_cubed`
     */
    void operator()(
        gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r_cubed,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::two_m_over_m_plus_r_cubed<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::spatial_metric_rr`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> spatial_metric_rr,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::spatial_metric_rr<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::one_over_spatial_metric_rr`
     */
    void operator()(
        gsl::not_null<Scalar<DataType>*> one_over_spatial_metric_rr,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::one_over_spatial_metric_rr<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::spatial_metric_rr_minus_f_0`
     */
    void operator()(
        gsl::not_null<Scalar<DataType>*> spatial_metric_rr_minus_f_0,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::spatial_metric_rr_minus_f_0<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::d_spatial_metric_rr`
     */
    void operator()(
        gsl::not_null<Scalar<DataType>*> d_spatial_metric_rr,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::d_spatial_metric_rr<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::d_f_0`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> d_f_0,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::d_f_0<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::d_f_0_times_x_over_r`
     */
    void operator()(
        gsl::not_null<tnsr::i<DataType, 3, Frame>*> d_f_0_times_x_over_r,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::d_f_0_times_x_over_r<DataType, Frame> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::f_1`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> f_1,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_1<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::f_1_times_x_over_r`
     */
    void operator()(
        gsl::not_null<tnsr::i<DataType, 3, Frame>*> f_1_times_x_over_r,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::f_1_times_x_over_r<DataType, Frame> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::f_2`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> f_2,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_2<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::f_2_times_xxx_over_r_cubed`
     */
    void operator()(
        gsl::not_null<tnsr::iii<DataType, 3, Frame>*>
            f_2_times_xxx_over_r_cubed,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::f_2_times_xxx_over_r_cubed<DataType, Frame> /*meta*/)
        const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::f_3`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> f_3,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_3<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by `internal_tags::f_4`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> f_4,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::f_4<DataType> /*meta*/) const;

    /*!
     * \brief Computes the lapse
     *
     * \details Computed as
     *
     * \f{align}
     *     \alpha &= \gamma_{rr}^{-1/2}
     * \f}
     *
     * where \f$\gamma_{rr}\f$ is a component of the spatial metric defined by
     * `internal_tags::spatial_metric_rr`.
     */
    void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                    gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::Lapse<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::neg_half_lapse_cubed_times_d_spatial_metric_rr`
     */
    void operator()(
        gsl::not_null<Scalar<DataType>*>
            neg_half_lapse_cubed_times_d_spatial_metric_rr,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::neg_half_lapse_cubed_times_d_spatial_metric_rr<
            DataType> /*meta*/) const;

    /*!
     * \brief Computes the shift
     *
     * \details Computed as
     *
     * \f{align}
     *     \beta^i &= \left(\frac{2M}{M+r}\right)^2 \frac{X^i}{r}
     *         \frac{1}{\gamma_{rr}}
     * \f}
     *
     * where \f$M\f$ is the mass of the black hole, \f$r\f$ is the radius
     * defined by `internal_tags::r`, \f$X^i\f$ is defined by
     * `internal_tags::x_minus_center`, and \f$\gamma_{rr}\f$ is defined by
     * `internal_tags::spatial_metric_rr`.
     */
    void operator()(gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
                    gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::Shift<DataType, 3, Frame> /*meta*/) const;

    /*!
     * \brief Computes the spatial derivative of the shift
     *
     * \details Computed as
     *
     * \f{align}
     *     \partial_k \beta^i &=
     *         f_4 \frac{X^i}{r} \frac{X_k}{r} + \delta_k^i f_3
     * \f}
     *
     * where \f$r\f$ is the radius defined by `internal_tags::r`, \f$X^i\f$ is
     * defined by `internal_tags::x_minus_center`, \f$X_j = X^i \delta_{ij}\f$,
     * \f$f_3\f$ is defined by `internal_tags::f_3`, and \f$f_4\f$ is defined by
     * `internal_tags::f_4`.
     */
    void operator()(gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
                    gsl::not_null<CachedBuffer*> cache,
                    DerivShift<DataType, Frame> /*meta*/) const;

    /*!
     * \brief Computes the spatial metric
     *
     * \details Computed as
     *
     * \f{align}
     *     \gamma_{ij} &=
     *         \left(\gamma_{rr} - \left(1+\frac{M}{r}\right)^2\right)
     *             \frac{X_i}{r} \frac{X_j}{r} +
     *         \delta_{ij} \left(1+\frac{M}{r}\right)^2
     * \f}
     *
     * where \f$M\f$ is the mass of the black hole, \f$r\f$ is the radius
     * defined by `internal_tags::r`, \f$\gamma_{rr}\f$ is defined by
     * `internal_tags::spatial_metric_rr`, and \f$X_j = X^i \delta_{ij}\f$ where
     * \f$X^i\f$ is defined by `internal_tags::x_minus_center`.
     */
    void operator()(gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
                    gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::SpatialMetric<DataType, 3, Frame> /*meta*/) const;

    /*!
     * \brief Computes the spatial derivative of the spatial metric
     *
     * \details Computed as
     *
     * \f{align}
     *     \partial_k \gamma_{ij} &=
     *         f_2 \frac{X_i}{r} \frac{X_j}{r} \frac{X_k}{r} +
     *         f_1 \frac{X_j}{r} \delta_{ik} + f_1 \frac{X_i}{r} \delta_{jk} +
     *         \partial_r f_0 \frac{X_k}{r} \delta_{ij}
     * \f}
     *
     * where \f$r\f$ is the radius defined by `internal_tags::r`,
     * \f$\partial_r f_0\f$ is defined by `internal_tags::d_f_0`, \f$f_1\f$ is
     * defined by `internal_tags::f_1`, \f$f_2\f$ is defined by
     * `internal_tags::f_2`, and \f$X_j = X^i \delta_{ij}\f$ where \f$X^i\f$ is
     * defined by `internal_tags::x_minus_center`.
     */
    void operator()(
        gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
        gsl::not_null<CachedBuffer*> cache,
        DerivSpatialMetric<DataType, Frame> /*meta*/) const;

    /*!
     * \brief Sets the time derivative of the spatial metric to 0
     */
    void operator()(
        gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
        gsl::not_null<CachedBuffer*> cache,
        ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>> /*meta*/) const;

    /*!
     * \brief Computes the determinant of the spatial metric
     */
    void operator()(gsl::not_null<Scalar<DataType>*> det_spatial_metric,
                    gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::DetSpatialMetric<DataType> /*meta*/) const;

    /*!
     * \brief Computes one over the determinant of the spatial metric
     */
    void operator()(
        gsl::not_null<Scalar<DataType>*> one_over_det_spatial_metric,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::one_over_det_spatial_metric<DataType> /*meta*/) const;

   private:
    /*!
     * \brief The harmonic Schwarzschild solution
     */
    const HarmonicSchwarzschild& solution_;
    /*!
     * \brief Cartesian coordinates of the position at which to compute
     * spacetime quantities
     */
    const tnsr::I<DataType, 3, Frame>& x_;
  };

  /*!
   * \brief Computes and returns spacetime quantities of interest
   */
  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateVars : public CachedBuffer<DataType, Frame> {
   public:
    using CachedBuffer = HarmonicSchwarzschild::CachedBuffer<DataType, Frame>;
    using CachedBuffer::CachedBuffer;
    using CachedBuffer::get_var;

    /*!
     * \brief Computes and returns the spatial derivative of the lapse
     *
     * \details Computed as
     *
     * \f{align}
     *     \partial_i \alpha &=
     *         -\frac{1}{2} \gamma_{rr}^{-3/2}
     *         \partial_r \gamma_{rr} \frac{X_i}{r}
     * \f}
     *
     * where \f$r\f$ is the radius defined by `internal_tags::r`,
     * \f$\gamma_{rr}\f$ is defined by `internal_tags::spatial_metric_rr`, its
     * derivative \f$\partial_r \gamma_{rr}\f$ is defined by
     * `internal_tags::d_spatial_metric_rr`, and \f$X_j = X^i \delta_{ij}\f$
     * where \f$X^i\f$ is defined by `internal_tags::x_minus_center`.
     */
    tnsr::i<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        DerivLapse<DataType, Frame> /*meta*/);

    /*!
     * \brief Returns the time derivative of the lapse, which is 0
     */
    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/);

    /*!
     * \brief Returns the time derivative of the shift, which is 0
     */
    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Shift<DataType, 3, Frame>> /*meta*/);

    /*!
     * \brief Computes and returns the square root of the determinant of the
     * spatial metric
     */
    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/);

    /*!
     * \brief Computes and returns the inverse spatial metric
     */
    tnsr::II<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::InverseSpatialMetric<DataType, 3, Frame> /*meta*/);

    /*!
     * \brief Computes and returns the extrinsic curvature
     */
    tnsr::ii<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::ExtrinsicCurvature<DataType, 3, Frame> /*meta*/);
  };

 private:
  /*!
   * \brief Mass of the black hole
   */
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  /*!
   * \brief Center of the black hole
   */
  std::array<double, volume_dim> center_ =
      make_array<volume_dim>(std::numeric_limits<double>::signaling_NaN());
};

/*!
 * \brief Return whether two harmonic Schwarzschild solutions are equivalent
 */
SPECTRE_ALWAYS_INLINE bool operator==(const HarmonicSchwarzschild& lhs,
                                      const HarmonicSchwarzschild& rhs) {
  return lhs.mass() == rhs.mass() and lhs.center() == rhs.center();
}

/*!
 * \brief Return whether two harmonic Schwarzschild solutions are not equivalent
 */
SPECTRE_ALWAYS_INLINE bool operator!=(const HarmonicSchwarzschild& lhs,
                                      const HarmonicSchwarzschild& rhs) {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace gr
