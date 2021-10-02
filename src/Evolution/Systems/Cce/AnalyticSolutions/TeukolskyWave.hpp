// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
class ComplexDataVector;
/// \endcond

namespace Cce {
namespace Solutions {

/*!
 * \brief Computes the analytic data for a Teukolsky wave solution described in
 * \cite Barkett2019uae.
 *
 * \details This test computes an outgoing perturbative wave solution in
 * spherical coordinates with wave profile
 *
 * \f[
 * F(u) = A e^{- u^2 / k^2}.
 * \f]
 */
struct TeukolskyWave : public SphericalMetricData {
  struct ExtractionRadius {
    using type = double;
    static constexpr Options::String help{
        "The extraction radius of the spherical solution"};
    static type lower_bound() { return 0.0; }
  };
  struct Amplitude {
    using type = double;
    static constexpr Options::String help{
        "The amplitude of the Teukolsky wave."};
    static type lower_bound() { return 0.0; }
  };
  struct Duration {
    using type = double;
    static constexpr Options::String help{
        "The characteristic duration of the Gaussian envelope."};
    static type lower_bound() { return 0.0; }
  };

  static constexpr Options::String help{
      "An analytic solution derived from the linearized Teukolsky equation"};

  using options = tmpl::list<ExtractionRadius, Amplitude, Duration>;

  WRAPPED_PUPable_decl_template(TeukolskyWave);  // NOLINT

  explicit TeukolskyWave(CkMigrateMessage* msg) : SphericalMetricData(msg) {}

  // clang doesn't manage to use = default correctly in this case
  // NOLINTNEXTLINE(modernize-use-equals-default)
  TeukolskyWave() {}

  TeukolskyWave(double extraction_radius, double amplitude, double duration);

  std::unique_ptr<WorldtubeData> get_clone() const override;

  void pup(PUP::er& p) override;

 private:
  /*
   * The A coefficient is calculated as
   *
   * \f[
   * A = 3 (3 k^4 + 4 r^2 u^2 - 2 k^2 r^2 (r + 3 u)) a e^{-u^2/k^2} / (k^4 r^5)
   * \f]
   *
   * where \f$r\f$ is the extraction radius, \f$u\f$ the retarded time, \f$a\f$
   * the amplitude of the pulse and \f$k\f$ the duration of the pulse.
   */
  double pulse_profile_coefficient_a(double time) const;

  /*
   * The B coefficient is calculated as
   *
   * \f[
   * B = 2 (-3 k^6 + 4 r^4 u^3 - 6 k^2 r^3 u (r + u) + 3 k^4 r^2 (r + 2 u))
   *  * a e^{-u^2/k^2} / (k^6 r^6)
   * \f]
   *
   * where \f$r\f$ is the extraction radius, \f$u\f$ the retarded time, \f$a\f$
   * the amplitude of the pulse and \f$k\f$ the duration of the pulse.
   */
  double pulse_profile_coefficient_b(double time) const;

  /*
   * The C coefficient is calculated as
   *
   * \f[
   * C = (1/4) (21 k^8 + 16 r^4 u^4 - 16 k^2 r^3 u^2 (3 r + u)
   * - 6 k^6 r (3 r + 7 u) + 12 k^4 r^2 (r^2 + 2 r u + 3 u^2))
   *  * a e^{-u^2/k^2} / (k^8 r^5)
   * \f]
   *
   * where \f$r\f$ is the extraction radius, \f$u\f$ the retarded time, \f$a\f$
   * the amplitude of the pulse and \f$k\f$ the duration of the pulse.
   */
  double pulse_profile_coefficient_c(double time) const;

  /*
   * The time derivative of the A coefficient is calculated as
   *
   * \f[
   * \partial_t A = -6 (4 r^2 u^3 + 3 k^4 (r + u) - 6 k^2 r u (r + u))
   *  a e^{-u^2/k^2} / (k^6 r^5)
   * \f]
   *
   * where \f$r\f$ is the extraction radius, \f$u\f$ the retarded time, \f$a\f$
   * the amplitude of the pulse and \f$k\f$ the duration of the pulse.
   */
  double dt_pulse_profile_coefficient_a(double time) const;

  /*
   * The time derivative of the B coefficient is calculated as
   *
   * \f[
   * \partial_t B = 4 (-4 r^4 u^4 + 6 k ^2 r^3 u^2 (2 r + u)
   * + 3 k^6 (r^2 + u) - 3 k^4 r^2 (r + u) (r + 2u))
   *  a e^{-u^2/k^2} / (k^8 r^6)
   * \f]
   *
   * where \f$r\f$ is the extraction radius, \f$u\f$ the retarded time, \f$a\f$
   * the amplitude of the pulse and \f$k\f$ the duration of the pulse.
   */
  double dt_pulse_profile_coefficient_b(double time) const;

  /*
   * The time derivative of the C coefficient is calculated as
   *
   * \f[
   * \partial_t C = (1/2) (-16 r^4 u^5 - 21 k^8 (r + u)
   * + 16 k^2 r^3 u^3 (5 r + u) + 6 k^6 r (r + u) (2 r + 7 u)
   * - 12 k^4 r^2 u (4 r^2 + 4 r u + 3 u^2))
   *  a e^{-u^2/k^2} / (k^10 r^5)
   * \f]
   *
   * where \f$r\f$ is the extraction radius, \f$u\f$ the retarded time, \f$a\f$
   * the amplitude of the pulse and \f$k\f$ the duration of the pulse.
   */
  double dt_pulse_profile_coefficient_c(double time) const;

  /*
   * The radial derivative of the A coefficient is calculated as
   *
   * \f[
   * \partial_r A + \partial_t A = - 9 (5 k^4 + 4 r^2 u^2 - 2 k^2 r (r + 4 u))
   * a e^{-u^2/k^2} / (k^4 r^6)
   * \f]
   *
   * where \f$r\f$ is the extraction radius, \f$u\f$ the retarded time, \f$a\f$
   * the amplitude of the pulse and \f$k\f$ the duration of the pulse.
   */
  double dr_pulse_profile_coefficient_a(double time) const;

  /*
   * The radial derivative of the B coefficient is calculated as
   *
   * \f[
   * \partial_r B + \partial_t B  = 2 (18 k^6 - 8 r^4 u^3 + 6 k^2 r^3 u (2 r + 3
   * u)
   * - 3 k^4 r^2 (3 r + 8 u)) a e^{-u^2/k^2} / (k^6 r^7)
   * \f]
   *
   * where \f$r\f$ is the extraction radius, \f$u\f$ the retarded time, \f$a\f$
   * the amplitude of the pulse and \f$k\f$ the duration of the pulse.
   */
  double dr_pulse_profile_coefficient_b(double time) const;

  /*
   * The radial derivative of the C coefficient is calculated as
   *
   * \f[
   * \partial_r C + \partial_t C  = -(1/4) (105 * k^8 + 16 r^4 u^4
   * - 16 k^2 r^3 u^2 (3 r + 2 u) - 6 k^6 r (9 r + 28 u)
   * + 12 k^4 r^2 (r^2 + 4 r u + 9 u^2))
   *  a e^{-u^2/k^2} / (k^8 r^6)
   * \f]
   *
   * where \f$r\f$ is the extraction radius, \f$u\f$ the retarded time, \f$a\f$
   * the amplitude of the pulse and \f$k\f$ the duration of the pulse.
   */
  double dr_pulse_profile_coefficient_c(double time) const;

  static DataVector sin_theta(size_t l_max);

  static DataVector cos_theta(size_t l_max);

 protected:
  /// A no-op as the Teukolsky wave solution does not have substantial
  /// shared computation to prepare before the separate component calculations.
  void prepare_solution(const size_t /*output_l_max*/,
                        const double /*time*/) const override {}

  /*!
   * \brief Compute the spherical coordinate metric from the closed-form
   * perturbative Teukolsky wave metric.
   *
   * \details The specific outgoing wave selected in this analytic solution is
   * constructed from a (2,0) mode as in \cite Barkett2019uae, and takes the
   * form
   *
   * \f{align*}{
   * g_{tt} &= -1\\
   * g_{rr} &= (1 + A f_{rr}) \\
   * g_{r \theta} &= 2 B f_{r \theta} r\\
   * g_{\theta \theta} &= (1 + C f_{\theta \theta}^{(C)}
   * + A f_{\theta \theta}^{(A)}) r^2\\
   * g_{\phi \phi} &= (1 + C f_{\phi \phi}^{(C)}
   * + A f_{\phi \phi}^{(A)}) r^2 \sin^2 \theta\\
   * \f}
   *
   * and all other components vanish. The angular factors generated by the
   * choice of spin-weighted spherical harmonic are
   *
   * \f{align*}{
   * f_{rr} &= 2 - 3 \sin^2 \theta \\
   * f_{r \theta} &=  -3 \sin \theta \cos \theta \\
   * f_{\theta \theta}^{(C)} &= 3 \sin^2 \theta \\
   * f_{\theta \theta}^{(A)} &= -1 \\
   * f_{\phi \phi}^{(C)} &= - 3 \sin^2 \theta \\
   * f_{\phi \phi}^{(A)} &= 3 \sin^2 \theta -1,
   * \f}
   *
   * the radial and time dependent factors are
   *
   * \f{align*}{
   * A &= 3 \left(\frac{\partial_u^2 F(u)}{r^3}
   * + \frac{3 \partial_u F(u)}{r^4} + \frac{3 F(u)}{r^5} \right),\\
   * B &= - \left(\frac{\partial_u^3 F(u)}{r^2}
   * + \frac{3 \partial_u^2 F(u)}{r^3} + \frac{6 \partial_uF(u)}{r^4}
   * + \frac{6 F(u)}{r^5}\right), \\
   * C &= \frac{1}{4} \left(\frac{\partial_u^4 F(u)}{r}
   * + \frac{2 \partial_u^3 F(u)}{r^2} + \frac{9 \partial_u^2 F(u)}{r^3}
   * + \frac{21 \partial_u F(u)}{r^4} + \frac{21 F(u)}{r}\right),
   * \f}
   *
   * and the pulse profile is
   *
   * \f[
   * F(u) = a e^{-u^2 /k^2}.
   * \f]
   *
   * So, the pulse profile factors expand to
   *
   * \f{align*}{
   * A &= \frac{3 a e^{-u^2/k^2}}{k^4 r^5} \left(3 k^4 + 4 r^2 u^2
   * - 2 k^2 r (r + 3 u)\right),\\
   * B &= \frac{2 a e^{-u^2/k^2}}{k^6 r^5} \left(-3 k^6 + 4 r^3 u^3
   * - 6 k^2 r^2 u (r + u) + 3 k^4 r (r + 2 u)\right), \\
   * C &= \frac{a e^{-u^2/k^2}}{4 k^8 r^5} \left(21 k^8 + 16 r^4 u^4
   * - 16 k^2 r^3 u^2 (3 r + u) - 6 k^6 r (3 r + 7 u)
   * + 12 k^4 r^2 (r^2 + 2 r u + 3 u^2)\right),
   * \f}
   *
   * \note The \f$\phi\f$ components are returned in a form for which the
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
      size_t l_max, double time) const override;

  /*!
   * \brief Compute the radial derivative of the spherical coordinate metric
   * from the closed-form perturbative Teukolsky wave metric.
   *
   * \details The specific outgoing wave selected in this analytic solution is
   * constructed from a (2,0) mode as in \cite Barkett2019uae, and takes the
   * form
   *
   * \f{align*}{
   * \partial_r g_{rr} &= f_{r r} \partial_r A \\
   * \partial_r g_{r \theta} &= f_{r \theta} (B + r \partial_r B)\\
   * \partial_r g_{\theta \theta} &= 2 (1 + C f_{\theta \theta}^{(C)}
   * + A f_{\theta \theta}^{(A)}) r
   * + (\partial_r C f_{\theta \theta}^{(C)}
   * + \partial_r A f_{\theta \theta}^{(A)}) r^2 \\
   * \partial_r g_{\phi \phi} &= 2 (1 + C f_{\phi \phi}^{(C)}
   * + A f_{\phi \phi}^{(A)}) r \sin^2 \theta
   * + (\partial_r C f_{\phi \phi}^{(C)}
   * + \partial_r A f_{\phi \phi}^{(A)}) r^2 \sin^2 \theta\\
   * \f}
   *
   * and all other components vanish. The angular factors \f$f_{a b}\f$ and the
   * metric component functions \f$A, B,\f$ and \f$C\f$ are defined as in
   * `TeukolskyWave::spherical_metric()`.
   * The radial derivatives of the pulse profile functions are obtained by:
   *
   * \f{align*}{
   * \partial_r A + \partial_t A &= \frac{-9 a e^{-u^2/k^2}}{k^4 r^6} \left(
   *  5 k^4 + 4 r^2 u^2 - 2 k^2 r (r + 4 u)\right), \\
   * \partial_r B + \partial_t B &= \frac{2 a e^{-u^2/k^2}}{k^6 r^6} \left(
   *  15 k^6 - 8 r^3 u^3 + 6 k^2 r^2 u (2 r + 3 u)
   *  - 3 k^4 r (3 r + 8 u)\right), \\
   * \partial_r C + \partial_t C &= \frac{-a e^{-u^2/k^2}}{4 k^8 r^6} \left(
   *  105 k^8 + 16 k^4 u^4 - 16 k^2 r^3 u^2 (3 r + 2 u) - 6 k^6 r (9 r + 28 u)
   *  + 12 k^4 r^2 (r^2 + 4 r u + 9 u^2)\right),
   * \f}
   *
   * and the time derivatives of the pulse profile functions are given in
   * `TeukolskyWave::dt_spherical_metric()`.
   *
   * \note The \f$\phi\f$ components are returned in a form for which the
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
      size_t l_max, double time) const override;

  /*!
   * \brief Compute the time derivative of the spherical coordinate metric
   * from the closed-form perturbative Teukolsky wave metric.
   *
   * \details The specific outgoing wave selected in this analytic solution is
   * constructed from a (2,0) mode as in \cite Barkett2019uae, and takes the
   * form
   *
   * \f{align*}{
   * \partial_t g_{rr} &= f_{r r} \partial_t A \\
   * \partial_t g_{r \theta} &= f_{r \theta} r \partial_t B\\
   * \partial_t g_{\theta \theta} &=
   * (\partial_t C f_{\theta \theta}^{(C)}
   * + \partial_t A f_{\theta \theta}^{(A)}) r^2 \\
   * \partial_t g_{\phi \phi} &= (\partial_t C f_{\phi \phi}^{(C)}
   * + \partial_t A f_{\phi \phi}^{(A)}) r^2 \sin^2 \theta\\
   * \f}
   *
   * and all other components vanish. The angular factors \f$f_{a b}\f$ and the
   * metric component functions \f$A, B,\f$ and \f$C\f$ are defined as in
   * `TeukolskyWave::spherical_metric()`.
   * The time derivatives of the pulse profile functions are:
   *
   * \f{align*}{
   * \partial_t A &=  \frac{-2 u}{k^2} A + \frac{3 a e^{-u^2/k^2}}{k^4 r^5}
   *  \left( 8 r^2 u - 6 k^2 r \right), \\
   * \partial_t B &= \frac{-2 u}{k^2} B + \frac{2 a e^{-u^2/k^2}}{k^6 r^5}
   *  \left(12 r^3 u^2 - 6 k^2 r^2 (r + 2 u) + 6 k^4 r\right), \\
   * \partial_t C &= \frac{-2 u}{k^2} C + \frac{-a e^{-u^2/k^2}}{4 k^8 r^5}
   * \left(64 k^4 u^3 - 16 k^2 r^3 u (6 r +  3 u) - 42 k^6 r
   *  + 12 k^4 r^2 (2 r + 6 u)\right),
   * \f}
   *
   * \note The \f$\phi\f$ components are returned in a form for which the
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
      size_t l_max, double time) const override;

  using WorldtubeData::variables_impl;

  using SphericalMetricData::variables_impl;

  /*!
   * \brief Compute the news associated with the (2,0)-mode Teukolsky wave.
   *
   * \details The value of the news is
   *
   * \f{align*}{
   * N = \frac{3 \sin^2 \theta}{4} \partial_u^5 F(u)
   * \f}
   *
   * where \f$F(u)\f$ is the pulse profile, taken to be
   *
   * \f[
   * F(u) = a e^{-u^2 /k^2},
   * \f]
   *
   * So, the news expands to
   *
   * \f[
   * N = -\frac{6 a e^{-u^2/k^2} u}{k^{10}} \left(15 k^4 - 20 k^2 u^2
   * + 4 u^4\right)
   * \f]
   *
   * in this analytic solution.
   */
  void variables_impl(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
      size_t l_max, double time,
      tmpl::type_<Tags::News> /*meta*/) const override;

  double amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double duration_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Solutions
}  // namespace Cce
