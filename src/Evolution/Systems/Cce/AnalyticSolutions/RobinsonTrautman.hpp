// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/History.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Solutions {
/*!
 * \brief An analytic solution representing a specialization of the radiative
 * Robinson-Trautman solution described in \cite Derry1970fk.
 *
 * \details This solution is not quite analytic, in the sense that there is a
 * single scalar field that must be evolved. Ultimately, it is a partial
 * specialization of the Characteristic equations such that \f$J = 0\f$ and the
 * evolution equations have been manipulated to give a time evolution equation
 * for \f$e^{-2 \beta}\f$, which is equivalent to the Robinson-Trautman scalar
 * \f$\omega_{\text{RT}}\f$ (denoted \f$W\f$ in \cite Derry1970fk -- we deviate
 * from their notation because the symbol \f$W\f$ is already used elsewhere in
 * the CCE system).
 *
 * \note The value of \f$\omega_{\text{RT}}\f$ should be real and near 1, which
 * is imposed by the solution itself, any modes specified in the input file are
 * treated as perturbations to the leading value of 1.0 for the
 * Robinson-Trautman scalar \f$\omega_{\text{RT}}\f$.
 */
struct RobinsonTrautman : public SphericalMetricData {
  struct InitialModes {
    using type = std::vector<std::complex<double>>;
    static constexpr Options::String help{
        "The initial modes of the Robinson-Trautman scalar, denoted W in "
        "[Derry 1970] and omega_RT in the rendered documentation. "
        "These are taken in ascending l and m order, m varies fastest. Note "
        "that the modes are treated as perturbations to the leading-order "
        "solution of 1.0 for omega_RT and only the real part of the field is "
        "used."};
  };
  struct ExtractionRadius {
    using type = double;
    static constexpr Options::String help{
        "The extraction radius of the spherical solution"};
    static type lower_bound() noexcept { return 0.0; }
    static type suggested_value() noexcept { return 20.0; }
  };
  struct LMax {
    using type = size_t;
    static constexpr Options::String help{
        "The maximum l value for the internal computation of the analytic "
        "solution"};
    static type lower_bound() noexcept { return 4; }
  };
  struct Tolerance {
    using type = double;
    static constexpr Options::String help{
        "The tolerance for the time evolution part of the calculation of the "
        "semi-analytic Robinson-Trautman solution"};
    static type lower_bound() noexcept { return 0.0; }
    static type suggested_value() noexcept { return 1.0e-11; }
  };
  struct StartTime {
    using type = double;
    static constexpr Options::String help{
        "The starting time for the Robinson-Trautman evolution"};
    static type lower_bound() noexcept { return 0.0; }
    static type suggested_value() noexcept { return 0.0; }
  };

  using options =
      tmpl::list<InitialModes, ExtractionRadius, LMax, Tolerance, StartTime>;

  static constexpr Options::String help = {
      "Analytic solution representing worldtube data for the nonlinear "
      "semi-analytic Robinson-Trautman metric, which requires a single "
      "scalar on the boundary to be evolved to determine the metric"};

  WRAPPED_PUPable_decl_template(RobinsonTrautman);  // NOLINT

  explicit RobinsonTrautman(CkMigrateMessage* msg) noexcept
      : SphericalMetricData(msg) {}

  RobinsonTrautman() = default;

  RobinsonTrautman(std::vector<std::complex<double>> initial_modes,
                   double extraction_radius, size_t l_max, double tolerance,
                   double start_time, const Options::Context& context);

  std::unique_ptr<WorldtubeData> get_clone() const noexcept override;

  void pup(PUP::er& p) noexcept override;

  bool use_noninertial_news() const noexcept override { return true; }

 private:
  void du_rt_scalar(
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> local_du_rt_scalar,
      const SpinWeighted<ComplexDataVector, 0>& rt_scalar) const noexcept;

  void du_bondi_w(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_bondi_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& local_du_rt_scalar,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_scalar)
      const noexcept;

  void bondi_u(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> bondi_u,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_scalar)
      const noexcept;

  void bondi_w(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_scalar)
      const noexcept;

  void dr_bondi_w(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> dr_bondi_w,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_scalar)
      const noexcept;

  void initialize_stepper_from_start() noexcept;

 protected:
  /*!
   * \brief The Robinson-Trautman solution performs the time-stepping to advance
   * the internal member scalar used to generate the metric solution to the
   * correct state for `time`.
   *
   * \details The generating scalar \f$\omega_{\text{RT}}\f$ is evolved
   * using equation (2.5) from \cite Derry1970fk (manipulated to a form
   * convenient for our numerical utilities)
   *
   * \f[
   * \partial_u \omega_{\text{RT}} = -
   * \left(\omega^4_{\text{RT}} \eth^2 \bar \eth^2 \omega_{\text{RT}}
   *  - \omega_{\text{RT}}^3 (\eth^2 \omega_{\text{RT}})
   * (\bar \eth^2  \omega_{\text{RT}}) \right)
   * \f]
   *
   * As the scalar \f$\omega_{\text{RT}}\f$ is evolved, it is filtered by
   * zeroing the highest two angular modes.
   */
  void prepare_solution(size_t output_l_max,
                        double time) const noexcept override;

  /*!
   * \brief Compute the spherical coordinate metric of the Robinson-Trautman
   * solution generated by the time-evolved scalar \f$\omega_{\text{RT}}\f$.
   *
   * \details The spacetime metric of the Robinson-Trautman solution can be
   * expressed as a specialization of the Bondi-Sachs metric (note the metric
   * signature change as compared to equation (1.2) from \cite Derry1970fk)
   *
   * \f[
   * ds^2 = -((r W + 1) \omega_{\text{RT}} - r^2 U \bar U) (dt - dr)^2
   * - 2 \omega_{\text{RT}} (dt - dr) dr
   * - 2 r^2 U^A q_{AB} dx^B (dt - dr) + r^2 q_{A B} dx^A dx^B,
   * \f]
   *
   * where \f$q_{A B}\f$ represents the angular unit sphere metric, and the
   * remaining Bondi-Sachs scalars and angular tensors are defined in terms of
   * the Robinson-Trautman scalar \f$\omega_{\text{RT}}\f$
   *
   * \f{align*}{
   * W &= \frac{1}{r}\left(\omega_{\text{RT}} + \eth \bar \eth
   * \omega_{\text{RT}} - 1\right) - \frac{2}{r^2 \omega_{\text{RT}}^2}\\
   * U &\equiv U^A q_A = \frac{\eth \omega_{\text{RT}}}{r}.
   * \f}
   * and \f$q_A\f$ is the angular dyad on the unit sphere.
   *
   * The angular part of the metric can be expressed in terms of the \f$U\f$
   * scalar as
   *
   * \f{align*}{
   * g_{u \theta} &= r^2 \Re U\\
   * g_{u \phi} &= r^2 \Im U
   * \f}
   */
  void spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          spherical_metric,
      size_t l_max, double time) const noexcept override;

  /*!
   * \brief Compute radial derivative of the spherical coordinate metric of the
   * Robinson-Trautman solution generated by the time-evolved scalar
   * \f$\omega_{\text{RT}}\f$.
   *
   * \details The radial derivative (at constant time t) of the
   * Robinson-Trautman solution is obtained by differentiating the expressions
   * from the documentation for `RobinsonTrautman::spherical_metric()`:
   *
   * \f{align*}{
   * (\partial_r g_{a b} + \partial_t g_{a b}) dx^a dx^b =
   * - (\omega_{\text{RT}} (r \partial_r W + W) - 2 r U \bar U
   * - r^2 (\bar U\partial_r U + U \partial_r \bar U)) (dt - dr)^2
   * - 2 r U^A q_{A B} dx^B (dt - dr) +  2 r q_{A B} dx^A dx^B
   * \f}
   *
   * where \f$q_{A B}\f$ represents the angular unit sphere metric, and the
   * remaining Bondi-Sachs scalars and angular tensors are defined in terms of
   * the Robinson-Trautman scalar \f$\omega_{\text{RT}}\f$
   *
   * \f{align*}{
   * W &= \frac{1}{r}\left(\omega_{\text{RT}}
   * + \eth \bar \eth \omega_{\text{RT}} - 1\right)
   * - \frac{2}{r^2 \omega_{\text{RT}}^2}\\
   * \partial_r W &= -\frac{1}{r^2} \left(\omega_{\text{RT}}
   * + \eth \bar \eth \omega_{\text{RT}} - 1\right)
   * + \frac{4}{r^3 \omega_{\text{RT}}^2}\\
   * U &\equiv U^A q_A = \frac{\eth \omega_{\text{RT}}}{r}.
   * \f}
   * and \f$q_A\f$ is the angular dyad on the unit sphere. The Robinson-Trautman
   * scalar \f$\omega_{\text{RT}}\f$ is independent of the Bondi radius \f$r\f$,
   * so all radial derivatives of \f$\omega_{\text{RT}}\f$ have been dropped
   */
  void dr_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dr_spherical_metric,
      size_t l_max, double time) const noexcept override;

  /*!
   * \brief Compute time derivative of the spherical coordinate metric of the
   * Robinson-Trautman solution generated by the time-evolved scalar
   * \f$\omega_{\text{RT}}\f$.
   *
   * \details The time derivative of the Robinson-Trautman solution is obtained
   * by differentiating the expressions from the documentation for
   * `RobinsonTrautman::spherical_metric()`:
   *
   * \f{align*}{
   * \partial_t g_{a b} dx^a dx^b =  -( \partial_u \omega_{\text{RT}} (r W + 1)
   *  + \omega_{\text{RT}} \partial_u W
   * - r^2 (\bar U \partial_u U + U \partial_u \bar U)) (dt - dr)^2
   * - 2 \partial_u \omega_{\text{RT}} (dt - dr) dr
   * - 2 r^2 \partial_u U^A q_{AB} dx^B (dt - dr),
   * \f}
   *
   * where \f$q_{A B}\f$ represents the angular unit sphere metric, and the
   * remaining Bondi-Sachs scalars and angular tensors are defined in terms of
   * the Robinson-Trautman scalar \f$\omega_{\text{RT}}\f$
   *
   * \f{align*}{
   * W &= \frac{1}{r}\left(\omega_{\text{RT}}
   * + \eth \bar \eth \omega_{\text{RT}} - 1\right)
   * - \frac{2}{r^2 \omega_{\text{RT}}^2}\\
   * \partial_u W &= \frac{1}{r}\left(\partial_u \omega_{\text{RT}}
   * + \eth \bar \eth \partial_u \omega_{\text{RT}}\right)
   * + \frac{4 \partial_u \omega_{\text{RT}}}{r^2 \omega_{\text{RT}}^3} \\
   * \partial_u U &= q_A \partial_u U^A = \frac{\eth \partial_u
   * \omega_{\text{RT}}}{r}, \f}
   *
   * and \f$q_A\f$ is the angular dyad on the unit sphere; and the time
   * derivative of the Robinson-Trautman scalar \f$\omega_{\text{RT}}\f$ is
   *
   * \f[
   * \partial_u \omega_{\text{RT}} =
   * \frac{1}{12} \left(-\omega^4_{\text{RT}} \eth^2 \bar \eth^2
   * \omega_{\text{RT}} + \omega_{\text{RT}}^3 (\eth^2 \omega_{\text{RT}})
   * (\bar \eth^2  \omega_{\text{RT}}) \right)
   * \f]
   */
  void dt_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dt_spherical_metric,
      size_t l_max, double time) const noexcept override;

  /*!
   * \brief Compute the news associated with the Robinson-Trautman solution
   * generated by the time-evolved scalar \f$\omega_{\text{RT}}\f$.
   *
   * \details The Bondi-Sachs news in the Robinson-Trautman solution is
   *
   * \f{align*}{
   * N = \frac{\bar \eth \bar \eth \omega_{\text{RT}}}{\omega_{\text{RT}}}
   * \f}
   */
  void variables_impl(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> News,
      size_t l_max, double time,
      tmpl::type_<Tags::News> /*meta*/) const noexcept override;

  using WorldtubeData::variables_impl;

  using SphericalMetricData::variables_impl;
 private:
  mutable std::pair<double, double> step_range_;

  mutable boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<
          boost::numeric::odeint::runge_kutta_dopri5<ComplexDataVector>>>
      stepper_;

  mutable Scalar<SpinWeighted<ComplexDataVector, 0>> dense_output_rt_scalar_;
  mutable Scalar<SpinWeighted<ComplexDataVector, 0>> dense_output_du_rt_scalar_;
  size_t l_max_ = 0;
  mutable double prepared_time_ = std::numeric_limits<double>::signaling_NaN();
  double tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double start_time_ = std::numeric_limits<double>::signaling_NaN();
  std::vector<std::complex<double>> initial_modes_;
};
}  // namespace Solutions
}  // namespace Cce
