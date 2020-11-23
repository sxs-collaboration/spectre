// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/StdComplex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
class ComplexDataVector;
/// \endcond

namespace Cce::Solutions {
namespace LinearizedBondiSachs_detail {
namespace InitializeJ {
// First hypersurface Initialization for the
// `Cce::Solutions::LinearizedBondiSachs` analytic solution.
//
// This initialization procedure should not be used except when the
// `Cce::Solutions::LinearizedBondiSachs` analytic solution is used,
// as a consequence, this initial data generator is deliberately not
// option-creatable; it should only be obtained from the `get_initialize_j`
// function of `Cce::InitializeJ::LinearizedBondiSachs`.
struct LinearizedBondiSachs : ::Cce::InitializeJ::InitializeJ {
  WRAPPED_PUPable_decl_template(LinearizedBondiSachs);  // NOLINT
  explicit LinearizedBondiSachs(CkMigrateMessage* /*unused*/) noexcept {}

  LinearizedBondiSachs() = default;

  LinearizedBondiSachs(double start_time, double frequency,
                       std::complex<double> c_2a, std::complex<double> c_2b,
                       std::complex<double> c_3a,
                       std::complex<double> c_3b) noexcept;

  std::unique_ptr<InitializeJ> get_clone() const noexcept override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points) const noexcept override;

  void pup(PUP::er& /*p*/) noexcept override;
 private:
  std::complex<double> c_2a_ = std::numeric_limits<double>::signaling_NaN();
  std::complex<double> c_2b_ = std::numeric_limits<double>::signaling_NaN();
  std::complex<double> c_3a_ = std::numeric_limits<double>::signaling_NaN();
  std::complex<double> c_3b_ = std::numeric_limits<double>::signaling_NaN();
  double frequency_ = std::numeric_limits<double>::signaling_NaN();
  double time_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace InitializeJ

// combine the (2,2) and (3,3) modes to collocation values for
// `bondi_quantity`
template <int Spin, typename FactorType>
void assign_components_from_l_factors(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> bondi_quantity,
    const FactorType& l_2_factor, const FactorType& l_3_factor, size_t l_max,
    double frequency, double time) noexcept;

// combine the (2,2) and (3,3) modes to time derivative collocation values for
// `bondi_quantity`
template <int Spin>
void assign_du_components_from_l_factors(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> du_bondi_quantity,
    const std::complex<double>& l_2_factor,
    const std::complex<double>& l_3_factor, size_t l_max, double frequency,
    double time) noexcept;
}  // namespace LinearizedBondiSachs_detail

/*!
 * \brief Computes the analytic data for a Linearized solution to the
 * Bondi-Sachs equations described in \cite Barkett2019uae.
 *
 * \details The solution represented by this function is generated with only
 * \f$(2,\pm2)\f$ and \f$(3,\pm3)\f$ modes, and is constructed according to the
 * linearized solution documented in Section VI of \cite Barkett2019uae. For
 * this solution, we impose additional restrictions that the linearized solution
 * be asymptotically flat so that it is compatible with the gauge
 * transformations performed in the SpECTRE regularity-preserving CCE. Using the
 * notation of \cite Barkett2019uae, we set:
 *
 * \f{align*}{
 * B_2 &= B_3 = 0\\
 * C_{2b} &= 3 C_{2a} / \nu^2\\
 * C_{3b} &= -3 i C_{3a} / \nu^3
 * \f}
 *
 * where \f$C_{2a}\f$ and \f$C_{3a}\f$ may be specified freely and are taken via
 * input option `InitialModes`.
 */
struct LinearizedBondiSachs : public SphericalMetricData {
  struct InitialModes {
    using type = std::array<std::complex<double>, 2>;
    static constexpr Options::String help{
        "The initial modes of the Robinson-Trautman scalar"};
  };
  struct ExtractionRadius {
    using type = double;
    static constexpr Options::String help{
        "The extraction radius of the spherical solution"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct Frequency {
    using type = double;
    static constexpr Options::String help{
        "The frequency of the linearized modes."};
    static type lower_bound() noexcept { return 0.0; }
  };

  static constexpr Options::String help{
    "A linearized Bondi-Sachs analytic solution"};

  using options = tmpl::list<InitialModes, ExtractionRadius, Frequency>;

  WRAPPED_PUPable_decl_template(LinearizedBondiSachs);  // NOLINT

  explicit LinearizedBondiSachs(CkMigrateMessage* /*unused*/) noexcept {}

  // clang doesn't manage to use = default correctly in this case
  // NOLINTNEXTLINE(hicpp-use-equals-default,modernize-use-equals-default)
  LinearizedBondiSachs() noexcept {}

  LinearizedBondiSachs(
      const std::array<std::complex<double>, 2>& mode_constants,
      double extraction_radius, double frequency) noexcept;

  std::unique_ptr<WorldtubeData> get_clone() const noexcept override;

  void pup(PUP::er& p) noexcept override;

  std::unique_ptr<Cce::InitializeJ::InitializeJ> get_initialize_j(
      double start_time) const noexcept override;

 protected:
  /// A no-op as the linearized solution does not have substantial shared
  /// computation to prepare before the separate component calculations.
  void prepare_solution(const size_t /*output_l_max*/,
                        const double /*time*/) const noexcept override {}

  /*!
   * \brief Computes the linearized solution for \f$J\f$.
   *
   * \details The linearized solution for \f$J\f$ is given by
   * \cite Barkett2019uae,
   *
   * \f[
   * J = \sqrt{12} ({}_2 Y_{2\,2} + {}_2 Y_{2\, -2})
   *  \mathrm{Re}(J_2(r) e^{i \nu u})
   * + \sqrt{60} ({}_2 Y_{3\,3} - {}_2 Y_{3\, -3})
   * \mathrm{Re}(J_3(r) e^{i \nu u}),
   * \f]
   *
   * where
   *
   * \f{align*}{
   * J_2(r) &= \frac{C_{2a}}{4 r} - \frac{C_{2b}}{12 r^3}, \\
   * J_3(r) &= \frac{C_{3a}}{10 r} - \frac{i \nu C_{3 b}}{6 r^3}
   * - \frac{C_{3 b}}{4 r^4}.
   * \f}
   */
  void linearized_bondi_j(
      gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> bondi_j, size_t l_max,
      double time) const noexcept;

  /*!
   * \brief Compute the linearized solution for \f$U\f$
   *
   * \details The linearized solution for \f$U\f$ is given by
   * \cite Barkett2019uae,
   *
   * \f[
   * U = \sqrt{3} ({}_1 Y_{2\,2} + {}_1 Y_{2\, -2})
   *  \mathrm{Re}(U_2(r) e^{i \nu u})
   * + \sqrt{6} ({}_1 Y_{3\,3} - {}_1 Y_{3\, -3})
   * \mathrm{Re}(U_3(r) e^{i \nu u}),
   * \f]
   *
   * where
   *
   * \f{align*}{
   * U_2(r) &= \frac{C_{2a}}{2 r^2} + \frac{i \nu C_{2 b}}{3 r^3}
   * + \frac{C_{2b}}{4 r^4} \\
   * U_3(r) &= \frac{C_{3a}}{2 r^2} - \frac{2 \nu^2 C_{3b}}{3 r^3}
   * + \frac{5 i \nu C_{3b}}{4 r^4} + \frac{C_{3 b}}{r^5}
   * \f}
   */
  void linearized_bondi_u(
      gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> bondi_u, size_t l_max,
      double time) const noexcept;

  /*!
   * \brief Computes the linearized solution for \f$W\f$.
   *
   * \details The linearized solution for \f$W\f$ is given by
   * \cite Barkett2019uae,
   *
   * \f[
   * W = \frac{1}{\sqrt{2}} ({}_0 Y_{2\,2} + {}_0 Y_{2\, -2})
   *  \mathrm{Re}(W_2(r) e^{i \nu u})
   * + \frac{1}{\sqrt{2}} ({}_0 Y_{3\,3} - {}_0 Y_{3\, -3})
   * \mathrm{Re}(W_3(r) e^{i \nu u}),
   * \f]
   *
   * where
   *
   * \f{align*}{
   * W_2(r) &= - \frac{\nu^2 C_{2b}}{r^2} + \frac{i \nu C_{2 b}}{r^3}
   * + \frac{C_{2b}}{2 r^4}, \\
   * W_3(r) &= -\frac{2 i \nu^3 C_{3b}}{r^2} - \frac{4 i \nu^2 C_{3b}}{r^3}
   * + \frac{5 \nu C_{3b}}{2 r^4} + \frac{3 C_{3b}}{r^5}.
   * \f}
   */
  void linearized_bondi_w(
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> bondi_w, size_t l_max,
      double time) const noexcept;

  /*!
   * \brief Computes the linearized solution for \f$\partial_r J\f$.
   *
   * \details The linearized solution for \f$\partial_r J\f$ is given by
   * \cite Barkett2019uae,
   *
   * \f[
   * \partial_r J = \sqrt{12} ({}_2 Y_{2\,2} + {}_2 Y_{2\, -2})
   *  \mathrm{Re}(\partial_r J_2(r) e^{i \nu u})
   * + \sqrt{60} ({}_2 Y_{3\,3} - {}_2 Y_{3\, -3})
   * \mathrm{Re}(\partial_r J_3(r) e^{i \nu u}),
   * \f]
   *
   * where
   *
   * \f{align*}{
   * \partial_r J_2(r) &= - \frac{C_{2a}}{4 r^2} + \frac{C_{2b}}{4 r^4}, \\
   * \partial_r J_3(r) &= -\frac{C_{3a}}{10 r^2} + \frac{i \nu C_{3 b}}{2 r^4}
   * + \frac{C_{3 b}}{r^5}.
   * \f}
   */
  void linearized_dr_bondi_j(
      gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> dr_bondi_j,
      size_t l_max, double time) const noexcept;

  /*!
   * \brief Compute the linearized solution for \f$\partial_r U\f$
   *
   * \details The linearized solution for \f$\partial_r U\f$ is given by
   * \cite Barkett2019uae,
   *
   * \f[
   * \partial_r U = \sqrt{3} ({}_1 Y_{2\,2} + {}_1 Y_{2\, -2})
   *  \mathrm{Re}(\partial_r U_2(r) e^{i \nu u})
   * + \sqrt{6} ({}_1 Y_{3\,3} - {}_1 Y_{3\, -3})
   * \mathrm{Re}(\partial_r U_3(r) e^{i \nu u}),
   * \f]
   *
   * where
   *
   * \f{align*}{
   * \partial_r U_2(r) &= -\frac{C_{2a}}{r^3} - \frac{i \nu C_{2 b}}{r^4}
   * - \frac{C_{2b}}{r^5} \\
   * \partial_r U_3(r) &= -\frac{C_{3a}}{r^3} + \frac{2 \nu^2 C_{3b}}{r^4}
   * - \frac{5 i \nu C_{3b}}{r^5} - \frac{5 C_{3 b}}{r^6}
   * \f}
   */
  void linearized_dr_bondi_u(
      gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> dr_bondi_u,
      size_t l_max, double time) const noexcept;

  /*!
   * \brief Computes the linearized solution for \f$\partial_r W\f$.
   *
   * \details The linearized solution for \f$W\f$ is given by
   * \cite Barkett2019uae,
   *
   * \f[
   * \partial_r W = \frac{1}{\sqrt{2}} ({}_0 Y_{2\,2} + {}_0 Y_{2\, -2})
   *  \mathrm{Re}(\partial_r W_2(r) e^{i \nu u})
   * + \frac{1}{\sqrt{2}} ({}_0 Y_{3\,3} - {}_0 Y_{3\, -3})
   * \mathrm{Re}(\partial_r W_3(r) e^{i \nu u}),
   * \f]
   *
   * where
   *
   * \f{align*}{
   * \partial_r W_2(r) &= \frac{2 \nu^2 C_{2b}}{r^3}
   * - \frac{3 i \nu C_{2 b}}{r^4} - \frac{2 C_{2b}}{r^5}, \\
   * \partial_r W_3(r) &= \frac{4 i \nu^3 C_{3b}}{r^3}
   * + \frac{12 i \nu^2  C_{3b}}{r^4} - \frac{10 \nu C_{3b}}{r^5}
   * - \frac{15 C_{3b}}{r^6}.
   * \f}
   */
  void linearized_dr_bondi_w(
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> dr_bondi_w,
      size_t l_max, double time) const noexcept;

  /*!
   * \brief Computes the linearized solution for \f$\partial_u J\f$.
   *
   * \details The linearized solution for \f$\partial_u J\f$ is given by
   * \cite Barkett2019uae,
   *
   * \f[
   * \partial_u J = \sqrt{12} ({}_2 Y_{2\,2} + {}_2 Y_{2\, -2})
   *  \mathrm{Re}(i \nu J_2(r)  e^{i \nu u})
   * + \sqrt{60} ({}_2 Y_{3\,3} - {}_2 Y_{3\, -3})
   * \mathrm{Re}(i \nu J_3(r) e^{i \nu u}),
   * \f]
   *
   * where
   *
   * \f{align*}{
   * J_2(r) &= \frac{C_{2a}}{4 r} - \frac{C_{2b}}{12 r^3}, \\
   * J_3(r) &= \frac{C_{3a}}{10 r} - \frac{i \nu C_{3 b}}{6 r^3}
   * - \frac{C_{3 b}}{4 r^4}.
   * \f}
   */
  void linearized_du_bondi_j(
      gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> du_bondi_j,
      size_t l_max, double time) const noexcept;

  /*!
   * \brief Compute the linearized solution for \f$\partial_u U\f$
   *
   * \details The linearized solution for \f$U\f$ is given by
   * \cite Barkett2019uae,
   *
   * \f[
   * \partial_u U = \sqrt{3} ({}_2 Y_{2\,2} + {}_2 Y_{2\, -2})
   *  \mathrm{Re}(i \nu U_2(r) e^{i \nu u})
   * + \sqrt{6} ({}_2 Y_{3\,3} - {}_2 Y_{3\, -3})
   * \mathrm{Re}(i \nu U_3(r) e^{i \nu u}),
   * \f]
   *
   * where
   *
   * \f{align*}{
   * U_2(r) &= \frac{C_{2a}}{2 r^2} + \frac{i \nu C_{2 b}}{3 r^3}
   * + \frac{C_{2b}}{4 r^4} \\
   * U_3(r) &= \frac{C_{3a}}{2 r^2} - \frac{2 \nu^2 C_{3b}}{3 r^3}
   * + \frac{5 i \nu C_{3b}}{4 r^4} + \frac{C_{3 b}}{r^5}
   * \f}
   */
  void linearized_du_bondi_u(
      gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> du_bondi_u,
      size_t l_max, double time) const noexcept;

  /*!
   * \brief Computes the linearized solution for \f$\partial_u W\f$.
   *
   * \details The linearized solution for \f$\partial_u W\f$ is given by
   * \cite Barkett2019uae,
   *
   * \f[
   * \partial_u W = \frac{1}{\sqrt{2}} ({}_1 Y_{2\,2} + {}_1 Y_{2\, -2})
   *  \mathrm{Re}(i \nu W_2(r) e^{i \nu u})
   * + \frac{1}{\sqrt{2}} ({}_1 Y_{3\,3} - {}_1 Y_{3\, -3})
   * \mathrm{Re}(i \nu W_3(r) e^{i \nu u}),
   * \f]
   *
   * where
   *
   * \f{align*}{
   * W_2(r) &= \frac{\nu^2 C_{2b}}{r^2} + \frac{i \nu C_{2 b}}{r^3}
   * + \frac{C_{2b}}{2 r^4}, \\
   * W_3(r) &= \frac{2 i \nu^3 C_{3b}}{r^2} - \frac{4 i \nu^2 C_{3b}}{r^3}
   * + \frac{5 \nu C_{3b}}{2 r^4} + \frac{3 C_{3b}}{r^5}.
   * \f}
   */
  void linearized_du_bondi_w(
      gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> du_bondi_w,
      size_t l_max, double time) const noexcept;

  /*!
   * \brief Compute the spherical coordinate metric from the linearized
   * Bondi-Sachs system.
   *
   * \details This function dispatches to the individual computations in this
   * class to determine the Bondi-Sachs scalars for the linearized solution.
   * Once the scalars are determined, the metric is assembled via (note
   * \f$\beta = 0\f$ in this solution)
   *
   * \f{align*}{
   * ds^2 =& - ((1 + r W) - r^2 h_{A B} U^A U^B) (dt - dr)^2
   * - 2  (dt - dr) dr \\
   * &- 2 r^2 h_{A B} U^B (dt - dr) dx^A + r^2 h_{A B} dx^A dx^B,
   * \f}
   *
   * where indices with capital letters refer to angular coordinates and the
   * angular tensors may be written in terms of spin-weighted scalars. Doing so
   * gives the metric components,
   *
   * \f{align*}{
   * g_{t t} &= -\left(1 + r W
   * - r^2 \Re\left(\bar J U^2 + K U \bar U\right)\right)\\
   * g_{t r} &= -1 - g_{t t}\\
   * g_{r r} &= 2 + g_{t t}\\
   * g_{t \theta} &= r^2 \Re\left(K U + J \bar U\right)\\
   * g_{t \phi} &= r^2 \Im\left(K U + J \bar U\right)\\
   * g_{r \theta} &= -g_{t \theta}\\
   * g_{r \phi} &= -g_{t \phi}\\
   * g_{\theta \theta} &= r^2 \Re\left(J + K\right)\\
   * g_{\theta \phi} &= r^2 \Im\left(J\right)\\
   * g_{\phi \phi} &= r^2 \Re\left(K - J\right),
   * \f}
   *
   * and all other components are zero.
   */
  void spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          spherical_metric,
      size_t l_max, double time) const noexcept override;

  /*!
   * \brief Compute the radial derivative of the spherical coordinate metric
   * from the linearized Bondi-Sachs system.
   *
   * \details This function dispatches to the individual computations in this
   * class to determine the Bondi-Sachs scalars for the linearized solution.
   * Once the scalars are determined, the radial derivative of the metric is
   * assembled via (note \f$\beta = 0\f$ in this solution)
   *
   * \f{align*}{
   * \partial_r g_{a b} dx^a dx^b =& - (W + r \partial_r W
   * - 2 r h_{A B} U^A U^B - r^2 (\partial_r h_{A B}) U^A U^B
   * - 2 r^2 h_{A B} U^A \partial_r U^B) (dt - dr)^2 \\
   * &- (4 r h_{A B} U^B + 2 r^2 ((\partial_r h_{A B}) U^B
   * + h_{AB} \partial_r U^B) ) (dt - dr) dx^A
   * + (2 r h_{A B} + r^2 \partial_r h_{A B}) dx^A dx^B,
   * \f}
   *
   * where indices with capital letters refer to angular coordinates and the
   * angular tensors may be written in terms of spin-weighted scalars. Doing so
   * gives the metric components,
   *
   * \f{align*}{
   * \partial_r g_{t t} &= -\left( W + r \partial_r W
   *  - 2 r \Re\left(\bar J U^2 + K U \bar U\right)
   * - r^2 \partial_r \Re\left(\bar J U^2 + K U \bar U\right)\right) \\
   * \partial_r g_{t r} &= -\partial_r g_{t t}\\
   * \partial_r g_{t \theta} &= 2 r \Re\left(K U + J \bar U\right)
   *   + r^2 \partial_r \Re\left(K U + J \bar U\right) \\
   * \partial_r g_{t \phi} &= 2r \Im\left(K U + J \bar U\right)
   *   + r^2 \partial_r \Im\left(K U + J \bar U\right) \\
   * \partial_r g_{r r} &= \partial_r g_{t t}\\
   * \partial_r g_{r \theta} &= -\partial_r g_{t \theta}\\
   * \partial_r g_{r \phi} &= -\partial_r g_{t \phi}\\
   * \partial_r g_{\theta \theta} &= 2 r \Re\left(J + K\right)
   *   + r^2 \Re\left(\partial_r J + \partial_r K\right) \\
   * \partial_r g_{\theta \phi} &= 2 r \Im\left(J\right)
   *  + r^2 \Im\left(\partial_r J\right)\\
   * \partial_r g_{\phi \phi} &= 2 r \Re\left(K - J\right)
   *  + r^2 \Re\left(\partial_r K - \partial_r J\right),
   * \f}
   *
   * and all other components are zero.
   */
  void dr_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dr_spherical_metric,
      size_t l_max, double time) const noexcept override;

  /*!
   * \brief Compute the time derivative of the spherical coordinate metric from
   * the linearized Bondi-Sachs system.
   *
   * \details This function dispatches to the individual computations in this
   * class to determine the Bondi-Sachs scalars for the linearized solution.
   * Once the scalars are determined, the metric is assembled via (note
   * \f$\beta = 0\f$ in this solution, and note that we take coordinate
   * \f$t=u\f$ in converting to the Cartesian coordinates)
   *
   * \f{align*}{
   * \partial_t g_{a b} dx^a dx^b =& - (r \partial_u W
   * - r^2 \partial_u h_{A B} U^A U^B
   * - 2 r^2 h_{A B} U^B \partial_u U^A) (dt - dr)^2 \\
   * &- 2 r^2 (\partial_u h_{A B} U^B + h_{A B} \partial_u U^B) (dt - dr) dx^A
   * + r^2 \partial_u h_{A B} dx^A dx^B,
   * \f}
   *
   * where indices with capital letters refer to angular coordinates and the
   * angular tensors may be written in terms of spin-weighted scalars. Doing so
   * gives the metric components,
   *
   * \f{align*}{
   * \partial_t g_{t t} &= -\left(r \partial_u W
   * - r^2 \partial_u \Re\left(\bar J U^2 + K U \bar U\right)\right)\\
   * \partial_t g_{t r} &= -\partial_t g_{t t}\\
   * \partial_t g_{t \theta} &= r^2 \partial_u \Re\left(K U + J \bar U\right)\\
   * \partial_t g_{t \phi} &= r^2 \partial_u \Im\left(K U + J \bar U\right)\\
   * \partial_t g_{r r} &= \partial_t g_{t t}\\
   * \partial_t g_{r \theta} &= -\partial_t g_{t \theta}\\
   * \partial_t g_{r \phi} &= -\partial_t g_{t \phi}\\
   * \partial_t g_{\theta \theta} &= r^2 \Re\left(\partial_u J
   *  + \partial_u K\right)\\
   * \partial_t g_{\theta \phi} &= r^2 \Im\left(\partial_u J\right)\\
   * \partial_t g_{\phi \phi} &= r^2 \Re\left(\partial_u K
   *  - \partial_u J\right),
   * \f}
   *
   * and all other components are zero.
   */
  void dt_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dt_spherical_metric,
      size_t l_max, double time) const noexcept override;

  using WorldtubeData::variables_impl;

  using SphericalMetricData::variables_impl;

  /*!
   * \brief Determines the News function from the linearized solution
   * parameters.
   *
   * \details The News is determined from the formula given in
   * \cite Barkett2019uae,
   *
   * \f{align*}{
   * N = \frac{1}{2 \sqrt{3}} ({}_{-2} Y_{2\, 2} + {}_{-2} Y_{2\,-2})
   * \Re\left(i \nu^3 C_{2 b} e^{i \nu u}\right)
   * + \frac{1}{\sqrt{15}} ({}_{-2} Y_{3\, 3} - {}_{-2} Y_{3\, -3})
   * \Re\left(- \nu^4 C_{3 b} e^{i \nu u} \right)
   * \f}
   */
  void variables_impl(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
      size_t l_max, double time, tmpl::type_<Tags::News> /*meta*/) const
      noexcept override;

  std::complex<double> c_2a_ = std::numeric_limits<double>::signaling_NaN();
  std::complex<double> c_3a_ = std::numeric_limits<double>::signaling_NaN();
  std::complex<double> c_2b_ = std::numeric_limits<double>::signaling_NaN();
  std::complex<double> c_3b_ = std::numeric_limits<double>::signaling_NaN();

  double frequency_ = 0.0;
};
}  // namespace Cce::Solutions
