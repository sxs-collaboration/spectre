// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
template <typename Tag>
struct dt;
template <typename Tag, typename Dim, typename Frame>
struct deriv;
}  // namespace Tags
/// \endcond

namespace gr::Solutions {

/*!
 * \brief Kerr initial data in the puncture radial coordinate of Liu, Etienne,
 * and Shapiro \cite Liu2009.
 *
 * \details
 * This class represents a Kerr black hole of mass \f$M\f$ and dimensionless
 * spin \f$\chi\f$ (with the spin along the \f$+z\f$ axis and the puncture at
 * the coordinate origin) in the puncture radial coordinate \f$r\f$ introduced
 * in \cite Liu2009. Defining the spin \f$a = \chi M\f$ and the horizon radii
 * \f$r_\pm = M \pm \sqrt{M^2 - a^2}\f$ (so that \f$r_+ r_- = a^2\f$ and
 * \f$r_+ + r_- = 2M\f$), the Boyer-Lindquist radius \f$r_\mathrm{BL}\f$ is
 * related to \f$r\f$ by
 *
 * \f{align}{
 * r_\mathrm{BL} = r\left(1 + \frac{r_+}{4r}\right)^2
 *              = r + \frac{r_+}{2} + \frac{r_+^2}{16 r} .
 * \f}
 *
 * The horizon (throat) sits at the coordinate radius \f$r = r_+/4\f$, which
 * stays finite (\f$\to M/4\f$) as \f$|\chi| \to 1\f$. The coordinate covers the
 * black hole exterior twice: the two sheets \f$r \gtrless r_+/4\f$ are joined
 * at the throat, and \f$r_\mathrm{BL} \ge r_+\f$ everywhere, so the interior
 * \f$r_\mathrm{BL} < r_+\f$ is never entered. Since the metric component
 * \f$\gamma_{rr}\f$ diverges at the throat like
 * \f$1/\sqrt{M^2 - a^2}\f$ as \f$|\chi| \to 1\f$ (the well-known infinite
 * proper throat of extremal Kerr), the class requires \f$|\chi| < 1\f$
 * strictly.
 *
 * The Boyer-Lindquist scalars are
 *
 * \f{align}{
 * \Sigma &= r_\mathrm{BL}^2 + \frac{a^2 z^2}{r^2}, \\
 * \Delta &= r_\mathrm{BL}^2 - 2 M r_\mathrm{BL} + a^2
 *         = (r_\mathrm{BL} - r_+)(r_\mathrm{BL} - r_-), \\
 * A &= (r_\mathrm{BL}^2 + a^2)^2
 *      - \Delta\, a^2 \left(1 - \frac{z^2}{r^2}\right) .
 * \f}
 *
 * The tensors are assembled directly in Cartesian form from the three mutually
 * orthogonal building blocks
 *
 * \f{align}{
 * n_i = \frac{x_i}{r}, \qquad
 * \lambda_i = (-y,\, x,\, 0), \qquad
 * \mu_i = (z x,\, z y,\, -(x^2 + y^2)) ,
 * \f}
 *
 * which are regular for \f$r > 0\f$ including on the spin axis, so no near-axis
 * special case is needed. The spatial metric and extrinsic curvature are
 *
 * \f{align}{
 * \gamma_{ij} &= c_\delta\, \delta_{ij} + c_n\, n_i n_j
 *              + c_\lambda\, \lambda_i \lambda_j, \\
 * K_{ij} &= c_{n\lambda}\, (n_i \lambda_j + n_j \lambda_i)
 *         + c_{\mu\lambda}\, (\mu_i \lambda_j + \mu_j \lambda_i),
 * \f}
 *
 * with the coefficient functions
 *
 * \f{align}{
 * c_\delta &= \frac{\Sigma}{r^2}, \qquad
 * c_n = \frac{\Sigma\, r_-}{r^2 (r_\mathrm{BL} - r_-)}, \qquad
 * c_\lambda = \frac{a^2 (\Sigma + 2 M r_\mathrm{BL})}{\Sigma\, r^4}, \\
 * c_{n\lambda} &= \frac{M a\, G\, \sqrt{r_\mathrm{BL}}}
 *   {\Sigma \sqrt{A \Sigma}\; r^3 \sqrt{r_\mathrm{BL} - r_-}}, \qquad
 * c_{\mu\lambda} = -\frac{2 a^3 M r_\mathrm{BL}\, z\, (r - r_+/4)}
 *   {\Sigma \sqrt{A \Sigma}\; r^6}\sqrt{\frac{r_\mathrm{BL} - r_-}{r}},
 * \f}
 *
 * where
 *
 * \f{align}{
 * G = 3 r_\mathrm{BL}^4 + 2 a^2 r_\mathrm{BL}^2 - a^4
 *     - a^2 (r_\mathrm{BL}^2 - a^2)\left(1 - \frac{z^2}{r^2}\right) .
 * \f}
 *
 * The stationary shift of \cite Liu2009 Eq. (7) is purely azimuthal,
 * \f$\beta^\phi = -2 M a r_\mathrm{BL}/A\f$, which in Cartesian form is
 *
 * \f{align}{
 * \beta^i = c_\beta\, \lambda^i, \qquad
 * c_\beta = -\frac{2 M a\, r_\mathrm{BL}}{A} .
 * \f}
 *
 * The analytic lapse
 *
 * \f{align}{
 * \alpha = \left(r - \frac{r_+}{4}\right) g(r, z) ,
 * \f}
 *
 * is negative on the inner sheet \f$r < r_+/4\f$. With this lapse the
 * returned representation is exactly stationary on both sheets, and all time
 * derivative tags vanish identically. Consumers that require an
 * everywhere-nonnegative initial lapse (e.g. moving-puncture evolutions)
 * must take the absolute value, recovering \cite Liu2009 Eq. (6).
 *
 * The inverse spatial metric and \f$\sqrt{\det\gamma}\f$ are returned from
 * their closed forms
 *
 * \f{align}{
 * \gamma^{ij} &= \frac{1}{c_\delta}\, \delta^{ij}
 *   - \frac{c_n}{c_\delta (c_\delta + c_n)}\, n^i n^j
 *   - \frac{c_\lambda}{c_\delta (c_\delta + \varpi^2 c_\lambda)}\,
 *     \lambda^i \lambda^j, \\
 * \sqrt{\det\gamma} &= \frac{1}{r^3}
 *   \sqrt{\frac{\Sigma\, r_\mathrm{BL}\, A}{r_\mathrm{BL} - r_-}},
 * \f}
 *
 * with \f$\varpi^2 = x^2 + y^2\f$, rather than by numerical inversion.
 *
 * Like gr::Solutions::TrumpetSchwarzschild, quantities diverge at the
 * puncture point \f$r = 0\f$ itself (the second asymptotically flat end); no
 * clamping is applied, so the origin must not coincide with a grid point.
 *
 * This solution reduces to the standard Schwarzschild solution in
 * isotropic coordinates when \f$\chi = 0\f$.
 *
 * The following input file options can be specified:
 *  - Mass (\f$M > 0\f$)
 *  - DimensionlessSpin (\f$\chi\f$, with \f$|\chi| < 1\f$ strictly)
 */
class HighSpinKerrPuncture : public MarkAsAnalyticSolution,
                             public AnalyticSolution<3_st> {
 private:
  template <typename DataType>
  struct IntermediateVars;

 public:
  static constexpr size_t volume_dim = 3;

  struct Mass {
    using type = double;
    static constexpr Options::String help = {"Mass of the Kerr black hole"};
    static type lower_bound() { return 0.; };
  };

  struct DimensionlessSpin {
    using type = double;
    static constexpr Options::String help = {
        "Dimensionless spin chi of the Kerr black hole, along the +z axis. "
        "Must satisfy |chi| < 1 strictly, since the metric degenerates at the "
        "throat in the extremal limit."};
  };

  using options = tmpl::list<Mass, DimensionlessSpin>;
  static constexpr Options::String help{
      "Kerr solution in the puncture radial coordinate of Liu, Etienne, and "
      "Shapiro (2009)."};

  HighSpinKerrPuncture(double mass, double dimensionless_spin,
                       const Options::Context& context = {});

  HighSpinKerrPuncture() = default;
  HighSpinKerrPuncture(const HighSpinKerrPuncture& /*rhs*/) = default;
  HighSpinKerrPuncture& operator=(const HighSpinKerrPuncture& /*rhs*/) =
      default;
  HighSpinKerrPuncture(HighSpinKerrPuncture&& /*rhs*/) = default;
  HighSpinKerrPuncture& operator=(HighSpinKerrPuncture&& /*rhs*/) = default;
  ~HighSpinKerrPuncture() = default;

  explicit HighSpinKerrPuncture(CkMigrateMessage* /*msg*/);

  template <typename DataType>
  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataType>,
                                   tmpl::size_t<volume_dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivShift = ::Tags::deriv<gr::Tags::Shift<DataType, volume_dim>,
                                   tmpl::size_t<volume_dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivSpatialMetric =
      ::Tags::deriv<gr::Tags::SpatialMetric<DataType, volume_dim>,
                    tmpl::size_t<volume_dim>, Frame::Inertial>;

  template <typename DataType>
  using tags = tmpl::list<
      gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
      DerivLapse<DataType>, gr::Tags::Shift<DataType, volume_dim>,
      ::Tags::dt<gr::Tags::Shift<DataType, volume_dim>>, DerivShift<DataType>,
      gr::Tags::SpatialMetric<DataType, volume_dim>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>,
      DerivSpatialMetric<DataType>, gr::Tags::SqrtDetSpatialMetric<DataType>,
      gr::Tags::ExtrinsicCurvature<DataType, volume_dim>,
      gr::Tags::InverseSpatialMetric<DataType, volume_dim>>;

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<Tags...> /*meta*/) const {
    const auto& vars =
        IntermediateVars<DataType>{mass_, dimensionless_spin_, x};
    return {get<Tags>(variables(x, t, vars, tmpl::list<Tags>{}))...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  SPECTRE_ALWAYS_INLINE double mass() const { return mass_; }
  SPECTRE_ALWAYS_INLINE double dimensionless_spin() const {
    return dimensionless_spin_;
  }

 private:
  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/)
      const -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<DerivLapse<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<DerivLapse<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<gr::Tags::Shift<DataType, volume_dim>> /*meta*/)
      const -> tuples::TaggedTuple<gr::Tags::Shift<DataType, volume_dim>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      const IntermediateVars<DataType>& vars,
      tmpl::list<::Tags::dt<gr::Tags::Shift<DataType, volume_dim>>> /*meta*/)
      const
      -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Shift<DataType, volume_dim>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<DerivShift<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<DerivShift<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      const IntermediateVars<DataType>& vars,
      tmpl::list<gr::Tags::SpatialMetric<DataType, volume_dim>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::SpatialMetric<DataType, volume_dim>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      const IntermediateVars<DataType>& vars,
      tmpl::list<
          ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>> /*meta*/)
      const -> tuples::TaggedTuple<
          ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<DerivSpatialMetric<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      const IntermediateVars<DataType>& vars,
      tmpl::list<gr::Tags::ExtrinsicCurvature<DataType, volume_dim>> /*meta*/)
      const -> tuples::TaggedTuple<
          gr::Tags::ExtrinsicCurvature<DataType, volume_dim>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      const IntermediateVars<DataType>& vars,
      tmpl::list<gr::Tags::InverseSpatialMetric<DataType, volume_dim>> /*meta*/)
      const -> tuples::TaggedTuple<
          gr::Tags::InverseSpatialMetric<DataType, volume_dim>>;

  // Intermediate quantities, computed once per call to variables().
  // Construct the radial building blocks r and z, the derived
  // constants r_+, r_-, the Boyer-Lindquist radius r_BL and its r-derivative,
  // the Boyer-Lindquist scalars Sigma, Delta, A, G and their (r, z) partials,
  // then the coefficient functions of the Cartesian assembly and their (r, z)
  // partials.
  template <typename DataType>
  struct IntermediateVars {
    IntermediateVars(double mass, double dimensionless_spin,
                     const tnsr::I<DataType, volume_dim, Frame::Inertial>& x);

    // Coordinate building blocks
    DataType r{};
    DataType z{};
    DataType one_over_r{};

    // Boyer-Lindquist radius and its r-derivative
    DataType r_bl{};
    DataType d_r_bl_d_r{};

    // Boyer-Lindquist scalars
    DataType sigma{};
    DataType delta{};
    DataType a_capital{};
    DataType g_capital{};
    DataType r_bl_minus_r_minus{};
    DataType r_minus_r_plus_over_four{};

    // (r, z) partials of the Boyer-Lindquist scalars
    DataType d_sigma_d_r{};
    DataType d_sigma_d_z{};
    DataType d_a_capital_d_r{};
    DataType d_a_capital_d_z{};
    DataType d_g_capital_d_r{};
    DataType d_g_capital_d_z{};

    // Cartesian coefficient functions and their (r, z) partials
    DataType c_delta{};
    DataType c_n{};
    DataType c_lambda{};
    DataType c_n_lambda{};
    // c_mu_lambda = (r - r_+/4) * z * c_mu_lambda_base. The two vanishing
    // factors (r - r_+/4) at the throat and z on the equator are kept
    // explicit so the coefficient and its derivatives avoid 0/0.
    DataType c_mu_lambda_base{};
    DataType c_mu_lambda{};
    DataType g_lapse{};
    DataType c_beta{};

    DataType d_c_delta_d_r{};
    DataType d_c_delta_d_z{};
    DataType d_c_n_d_r{};
    DataType d_c_n_d_z{};
    DataType d_c_lambda_d_r{};
    DataType d_c_lambda_d_z{};
    DataType d_c_n_lambda_d_r{};
    DataType d_c_n_lambda_d_z{};
    DataType d_c_mu_lambda_d_r{};
    DataType d_c_mu_lambda_d_z{};
    DataType d_g_lapse_d_r{};
    DataType d_g_lapse_d_z{};
    DataType d_c_beta_d_r{};
    DataType d_c_beta_d_z{};

    // Derived scalar constants of the black hole
    double mass;
    double spin_a;
    double r_plus;
    double r_minus;
  };

  double mass_{std::numeric_limits<double>::signaling_NaN()};
  double dimensionless_spin_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator==(const HighSpinKerrPuncture& lhs,
                const HighSpinKerrPuncture& rhs);
bool operator!=(const HighSpinKerrPuncture& lhs,
                const HighSpinKerrPuncture& rhs);
}  // namespace gr::Solutions
