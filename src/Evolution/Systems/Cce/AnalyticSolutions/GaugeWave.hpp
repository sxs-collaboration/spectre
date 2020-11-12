// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
class ComplexDataVector;
/// \endcond

namespace Cce {
namespace Solutions {

/*!
 * \brief Computes the analytic data for a gauge wave solution described in
 * \cite Barkett2019uae.
 *
 * \details This test computes an analytic solution of a pure-gauge perturbation
 * of the Schwarzschild metric. The gauge perturbation is constructed using the
 * time-dependent coordinate transformation of the ingoing Eddington-Finklestein
 * coordinate \f$\nu \rightarrow \nu + F(t - r) / r\f$, where
 *
 * \f[
 * F(u) = A \sin(\omega u) e^{- (u - u_0)^2 / k^2}.
 * \f]
 *
 * \note In the paper \cite Barkett2019uae, a translation map was
 * applied to the solution to make the test more demanding. For simplicity, we
 * omit that extra map. The behavior of translation-independence is
 * tested by the `Cce::Solutions::BouncingBlackHole` solution.
 */
struct GaugeWave : public SphericalMetricData {
  struct ExtractionRadius {
    using type = double;
    static constexpr Options::String help{
        "The extraction radius of the spherical solution"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct Mass {
    using type = double;
    static constexpr Options::String help{
        "The mass of the Schwarzschild solution."};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct Frequency {
    using type = double;
    static constexpr Options::String help{
        "The frequency of the oscillation of the gauge wave."};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct Amplitude {
    using type = double;
    static constexpr Options::String help{
      "The amplitude of the gauge wave."};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct PeakTime {
    using type = double;
    static constexpr Options::String help{
        "The time of the peak of the Gaussian envelope."};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct Duration {
    using type = double;
    static constexpr Options::String help{
        "The characteristic duration of the Gaussian envelope."};
    static type lower_bound() noexcept { return 0.0; }
  };

  using options = tmpl::list<ExtractionRadius, Mass, Frequency, Amplitude,
                             PeakTime, Duration>;

  static constexpr Options::String help = {
      "Analytic solution representing worldtube data for a pure-gauge "
      "perturbation near a Schwarzschild metric in spherical coordinates"};

  WRAPPED_PUPable_decl_template(GaugeWave);  // NOLINT

  explicit GaugeWave(CkMigrateMessage* /*unused*/) noexcept {}

  // clang doesn't manage to use = default correctly in this case
  // NOLINTNEXTLINE(modernize-use-equals-default)
  GaugeWave() noexcept {}

  GaugeWave(double extraction_radius, double mass, double frequency,
            double amplitude, double peak_time, double duration) noexcept;

  std::unique_ptr<WorldtubeData> get_clone() const noexcept override;
 private:
  double coordinate_wave_function(double time) const noexcept;

  double du_coordinate_wave_function(double time) const noexcept;

  double du_du_coordinate_wave_function(double time) const noexcept;

 protected:
  /// A no-op as the gauge wave solution does not have substantial
  /// shared computation to prepare before the separate component calculations.
  void prepare_solution(const size_t /*output_l_max*/,
                        const double /*time*/) const noexcept override {}

  /*!
   * \brief Compute the spherical coordinate metric from the closed-form gauge
   * wave metric.
   *
   * \details The transformation of the ingoing Eddington-Finkelstein coordinate
   * produces metric components in spherical coordinates (identical up to minor
   * manipulations of the metric given in Eq. (149) of \cite Barkett2019uae):
   *
   * \f{align*}{
   * g_{tt} &= \frac{-1}{r^3}\left(r - 2 M\right)
   * \left[r + \partial_u F(u)\right]^2\\
   * g_{rt} &= \frac{1}{r^4} \left[r + \partial_u F(u)\right]
   * \left\{2 M r^2 + \left(r - 2 M\right)
   * \left[r \partial_u F(u) + F(u)\right]\right\} \\
   * g_{rr} &= \frac{1}{r^5} \left[r^2 - r \partial_u F(u) - F(u)\right]
   * \left\{r^3 + 2 M r^2 + \left(r - 2 M\right)
   * \left[r \partial_u F(u) + F(u)\right]\right\} \\
   * g_{\theta \theta} &= r^2 \\
   * g_{\phi \phi} &= r^2 \sin^2(\theta),
   * \f}
   *
   * and all other components vanish. Here, \f$F(u)\f$ is defined as
   *
   * \f{align*}{
   * F(u) &= A \sin(\omega u) e^{-(u - u_0)^2 /k^2},\\
   * \partial_u F(u) &= A \left[-2 \frac{u - u_0}{k^2} \sin(\omega u)
   * + \omega \cos(\omega u)\right] e^{-(u - u_0)^2 / k^2}.
   * \f}
   *
   * \warning The \f$\phi\f$ components are returned in a form for which the
   * \f$\sin(\theta)\f$ factors are omitted, assuming that derivatives and
   * Jacobians will be applied similarly omitting those factors (and therefore
   * improving precision of the tensor expression). If you require the
   * \f$\sin(\theta)\f$ factors, be sure to put them in by hand in the calling
   * code.
   */
  void spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          spherical_metric,
      size_t l_max, double time) const noexcept override;

  /*!
   * \brief Compute the radial derivative of the spherical coordinate metric
   * from the closed-form gauge wave metric.
   *
   * \details The transformation of the ingoing Eddington-Finkelstein coordinate
   * produces the radial derivative of the metric components in spherical
   * coordinates:
   *
   * \f{align*}{
   * \partial_r g_{tt} + \partial_t g_{tt} =& \frac{2}{r^4}
   * \left[r + \partial_u F(u)\right]
   * \left[- M r + (r - 3 M)\partial_u F(u)\right] \\
   * \partial_r g_{rt} + \partial_t g_{rt} =& - \frac{1}{r^5}
   * \left\{2 M r^3 + 2 r F(u) (r - 3M) + \partial_u F(u)[r^3 + F(u)(3r - 8 M)]
   * + 2 r [\partial_u F(u)]^2 (r - 3M)\right\}\\
   * \partial_r g_{rr} + \partial_t g_{rr} =& \frac{2}{r^6}
   * \left\{- M r^4 + F(u)^2 (2r - 5M)
   * + \partial_u F(u) r^2 \left[4 M r + \partial_u F(u) (r - 3 M)\right]
   * + F(u) r \left[6 M r + \partial_u F(u) (3r - 8 M)\right]\right\} \\
   * g_{\theta \theta} =& 2 r \\
   * g_{\phi \phi} =& 2 r \sin^2(\theta),
   * \f}
   *
   * and all other components vanish (these formulae are obtained simply by
   * applying radial derivatives to those given in
   * `GaugeWave::spherical_metric()`). Here, \f$F(u)\f$ is defined as
   *
   * \f{align*}{
   * F(u) &= A \sin(\omega u) e^{-(u - u_0)^2 /k^2},\\
   * \partial_u F(u) &= A \left[-2 \frac{u - u_0}{k^2} \sin(\omega u)
   * + \omega \cos(\omega u)\right] e^{-(u - u_0)^2 / k^2}.
   * \f}
   *
   * \warning The \f$\phi\f$ components are returned in a form for which the
   * \f$\sin(\theta)\f$ factors are omitted, assuming that derivatives and
   * Jacobians will be applied similarly omitting those factors (and therefore
   * improving precision of the tensor expression). If you require the
   * \f$\sin(\theta)\f$ factors, be sure to put them in by hand in the calling
   * code.
   */
  void dr_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dr_spherical_metric,
      size_t l_max, double time) const noexcept override;

  /*!
   * \brief Compute the spherical coordinate metric from the closed-form gauge
   * wave metric.
   *
   * \details The transformation of the ingoing Eddington-Finkelstein coordinate
   * produces metric components in spherical coordinates:
   *
   * \f{align*}{
   * \partial_t g_{tt} =& \frac{-2 \partial_u^2 F(u)}{r^3}
   * \left(r - 2 M\right) \left(r + \partial_u F(u)\right) \\
   * \partial_t g_{rt} =& \frac{1}{r^4} \Bigg\{\partial_u^2 F(u)
   * \left[2 M r^2 + \left(r - 2 M\right)
   * (r \partial_u F(u) + F(u))\right] \\
   * &+ \left[r + \partial_u F(u)\right]
   * \left(r - 2M\right) \left[r \partial_u^2 F(u) + \partial_u F(u)
   * \right]\Bigg\} \\
   * \partial_t g_{rr} =&
   * \frac{1}{r^5}\Bigg\{-\left[r \partial_u^2 F(u) + \partial_u F(u)\right]
   * \left[r^3 + 2 M r^2  + \left(r - 2 M\right)
   * \left(r \partial_u F(u) + F(u)\right)\right]\\
   * &+ \left[r^2 - r \partial_u F(u) - F(u)\right]
   * \left(r - 2 M\right)
   * \left[r \partial_u^2 F(u) + \partial_u F(u)\right]\Bigg\} \\
   * \partial_t g_{\theta \theta} =& 0 \\
   * \partial_t g_{\phi \phi} =& 0,
   * \f}
   *
   * and all other components vanish. Here, \f$F(u)\f$ is defined as
   *
   * \f{align*}{
   * F(u) &= A \sin(\omega u) e^{-(u - u_0)^2 /k^2},\\
   * \partial_u F(u) &= A \left[-2 \frac{u - u_0}{k^2} \sin(\omega u)
   * + \omega \cos(\omega u)\right] e^{-(u - u_0)^2 / k^2},\\
   * \partial^2_u F(u) &= \frac{A}{k^4} \left\{-4 k^2 \omega (u - u_0)
   * \cos(\omega u) + \left[-2 k^2 + 4 (u - u_0)^2 - k^4 \omega^2\right]
   * \sin(\omega u)\right\}  e^{-(u - u_0) / k^2}
   * \f}
   *
   * \warning The \f$\phi\f$ components are returned in a form for which the
   * \f$\sin(\theta)\f$ factors are omitted, assuming that derivatives and
   * Jacobians will be applied similarly omitting those factors (and therefore
   * improving precision of the tensor expression). If you require the
   * \f$\sin(\theta)\f$ factors, be sure to put them in by hand in the calling
   * code.
   */
  void dt_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dt_spherical_metric,
      size_t l_max, double time) const noexcept override;

  using WorldtubeData::variables_impl;

  using SphericalMetricData::variables_impl;

  /// The News vanishes, because the wave is pure gauge.
  void variables_impl(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> News,
      size_t l_max, double time,
      tmpl::type_<Tags::News> /*meta*/) const noexcept override;

  double mass_ = std::numeric_limits<double>::signaling_NaN();
  double frequency_ = std::numeric_limits<double>::signaling_NaN();
  double amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double peak_time_ = std::numeric_limits<double>::signaling_NaN();
  double duration_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Solutions
}  // namespace Cce
