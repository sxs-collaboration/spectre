// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "NumericalAlgorithms/Interpolation/CubicSpline.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

namespace gr::Solutions {

/*!
 * \brief Trumpet Schwarzschild solution in isotropic coordinates
 *
 * \details
 * This solution is a trumpet Schwarzschild black hole in the isotropic
 * coordinates. It is a time-independent puncture solution in 1+log slicing.
 * The solution cannot be written down analytically in Schwarzschild
 * coordinates or isotropic coordinates. Refer to \cite Hannam2008
 * for equations and details.
 *
 * We first set up a source grid in isotropic radial coordinate r, on which
 * we compute the lapse and Schwarzschild radial coordinate R. The lapse
 * is computed by solving eq. (54)-(56) using the toms748 algorithm, where
 * the integrals in eq. (54)-(56) are evaluated by a tanh_sinh integrator.
 *
 * Eq. (54) is for \f$\alpha < \alpha_s=0.1\f$:
 * \f{equation*}{
 * r(\alpha) = R(\alpha_s)^{(1/\alpha_s)} \exp \left[ -
 * \int_{\alpha}^{\alpha_s} \frac{1}{\bar{\alpha} R(\bar{\alpha})}
 * \frac{dR}{d\alpha}(\bar{\alpha}) \: d\bar{\alpha} - C_0 \right],
 * \f}
 * where eq. (55) gives
 * \f{equation*}{
 * C_0 = \int_{\alpha_s}^{1} \frac{\ln R(\alpha)}{\alpha^2} d\alpha.
 * \f}
 * Eq. (56) is for \f$\alpha > \alpha_s=0.1\f$:
 * \f{equation*}{
 * r(\alpha) = R(\alpha)^{(1/\alpha)} \exp \left[
 * \int_{\alpha_s}^{\alpha} \frac{\ln R(\bar{\alpha})}{\bar{\alpha}^2}
 * d\bar{\alpha} - C_0 \right].
 * \f}
 *
 * A standard Gauss quadratures will fail due to singular
 * behaviors of the integrands near the integration limits. The Schwarzschild R
 * is then computed by solving eq. (39) using again the toms748 algorithm.
 * \f{equation*}{
 * \alpha^2 = 1 - \frac{2M}{R} + \frac{C(n)^2 e^{2 \alpha / n}
 * }{R^4}.
 * \f}
 * Note that for a value of n near 2 (corresponding to standard 1+log slicing),
 * eq. (39) has two positive roots for a fixed lapse in [0., 1.).
 * See the following plot by Mathematica. We choose
 * \image html trumpet_two_branches_illustration.png
 * the physical root, i.e. with the correct asymptotic behaviors: R diverges as
 * \f$\alpha\f$ tends to 1, and R tends to the smaller solution as \f$\alpha\f$
 * tends to 0.

 * The physical root is numerically selected by finding the
 * critical lapse and critical Schwarzschild R, below which we tell the toms748
 * solver to find a root \f$R\in\f$ [min_schwarzschild_r, crit_schwarzschild_r]
 * or else we find a root
 * \f$R\in\f$ [crit_schwarzschild_r, max_schwarzschild_r]. min_schwarzschild_r
 * is currently selected to be 0., and max_schwarzschild_r is at least as large
 * as max_isotropic_r (currently 5000M), the maximum coordinate radius we
 * support for this initial data. In case the solver is asked to find a
 * Schwarzschild R greater than the max_isotropic_r, we use double the
 * asymptotic solution from eq. (39), i.e. \f$4/(1-\alpha^2)\f$, as solver
 * upper bound. This latter upper bound is necessary since the integrator
 * in eq. (55) needs lapse very close to 1 to converge.
 *
 * After acquiring the lapse and Schwarzschild R, we can assemble all the 3+1
 * quantities in the isotropic coordinates on the source grid.
 *
 * Since the user supplies grid points in the Cartesian version of the
 * isotropic coordinates, we compute a user grid in the isotropic radial
 * coordinate based on the Cartesian grid, compute the lapse and Schwarzschild
 * R on the source grid, interpolate to the user grid, and then assemble all
 * 3+1 quantities on the user grid and transform them back to the Cartesian
 * grid.
 *
 * To insulate our implementation from different mass parameters, we
 * nondimensionalize the above process using the black hole mass until
 * the final step of computing 3+1 quantities on the Cartesian grid,
 * where we restore the correct unit.
 *
 * The following are input file options that can be specified:
 *  - Mass
 *  - N (the parameter n in the slicing condition eq. (36))
 *
 * \note
 * N=2. is strongly suggested, as this gives a stationary solution
 * in the standard 1+log slicing. The other values of N near 2.
 * should work but have not been tested thoroughly. N outside of
 * [2., 3.] has not been tested at all.
 *
 * Some quantities very close to the puncture (<1.e-4 in isotropic radius)
 * may have larger truncation errors. This is expected since some quantities
 * such as the determiant of the spatial metric diverges at the puncture.
 */
class TrumpetSchwarzschild : public MarkAsAnalyticSolution,
                             public AnalyticSolution<3_st> {
 private:
  template <typename DataType>
  struct IntermediateVars;

 public:
  static constexpr size_t volume_dim = 3;

  struct Mass {
    using type = double;
    static constexpr Options::String help = {
        "Mass of the Schwarzschild black hole"};
    static type lower_bound() { return 0.1; };
  };

  // currently we have only tested around N=2; for other N eq. (39) may need to
  // be solved differently
  struct N {
    using type = double;
    static constexpr Options::String help = {
        "Parameter of the trumpet solution family. N=2. gives"
        "a stationary solution in standard 1+log slicing."
        "An N value outside [2., 3.] has not been tested."};
    static type lower_bound() { return 0.; };
  };

  using options = tmpl::list<Mass, N>;
  static constexpr Options::String help{
      "Schwarzschild solution in trumpet isotropic coordinates"};

  TrumpetSchwarzschild(double mass, double n,
                       const Options::Context& context = {});

  TrumpetSchwarzschild() = default;
  TrumpetSchwarzschild(const TrumpetSchwarzschild& /*rhs*/) = default;
  TrumpetSchwarzschild& operator=(const TrumpetSchwarzschild& /*rhs*/) =
      default;
  TrumpetSchwarzschild(TrumpetSchwarzschild&& /*rhs*/) = default;
  TrumpetSchwarzschild& operator=(TrumpetSchwarzschild&& /*rhs*/) = default;
  ~TrumpetSchwarzschild() = default;

  explicit TrumpetSchwarzschild(CkMigrateMessage* /*msg*/);

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

  // The user input Cartesian "x" is in isotropic coordinates
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<Tags...> /*meta*/) const {
    const auto& vars =
        IntermediateVars<DataType>{mass_, n_, x, t, data_on_source_grid_};
    return {get<Tags>(variables(x, t, vars, tmpl::list<Tags>{}))...};
  }

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      const IntermediateVars<DataType>& vars,
      tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1,
                  "Unrecognized tag requested.  See the function parameters "
                  "for the tag.");
    return {get<Tags>(variables(x, t, vars, tmpl::list<Tags>{}))...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  SPECTRE_ALWAYS_INLINE double mass() const { return mass_; }
  SPECTRE_ALWAYS_INLINE double n() const { return n_; }

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
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
                 double /*t*/, const IntermediateVars<DataType>& vars,
                 tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<DerivSpatialMetric<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
                 double /*t*/, const IntermediateVars<DataType>& vars,
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

  template <typename DataType>
  struct IntermediateVars {
    IntermediateVars(double mass, double n,
                     const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                     double /*t*/,
                     const std::array<DataVector, 2>& data_on_source_grid);
    // the lapse, Schwarzschild R, and isotropic r on isotropic grid points
    DataType lapse_on_user_grid{};
    DataType schwarzschild_r_on_user_grid{};
    DataType d_lapse_d_schwarzschild_r_on_user_grid{};
    DataType isotropic_r_on_user_grid{};

    // cached value to avoid division
    DataType one_over_schwarzschild_r_on_user_grid{};
    DataType one_over_isotropic_r_on_user_grid{};
    double one_over_mass;
    double one_over_n;
  };

  double mass_{std::numeric_limits<double>::signaling_NaN()};
  double n_{std::numeric_limits<double>::signaling_NaN()};

  const static tnsr::I<DataVector, 1, Frame::Inertial> source_grid_;
  std::array<DataVector, 2> data_on_source_grid_;
};

bool operator==(const TrumpetSchwarzschild& lhs,
                const TrumpetSchwarzschild& rhs);
bool operator!=(const TrumpetSchwarzschild& lhs,
                const TrumpetSchwarzschild& rhs);
}  // namespace gr::Solutions
