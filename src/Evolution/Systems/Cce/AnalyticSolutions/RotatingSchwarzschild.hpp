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

namespace Cce::Solutions {

/*!
 * \brief Computes the analytic data for the rotating Schwarzschild solution
 * described in \cite Barkett2019uae, section VI.C.
 *
 * \details This is a comparatively simple test which simply determines the
 * Schwarzschild metric in transformed coordinates given by \f$\phi\rightarrow
 * \phi + \omega u\f$, where \f$u\f$ is the retarded time.
 */
struct RotatingSchwarzschild : public SphericalMetricData {
  struct ExtractionRadius {
    using type = double;
    static constexpr Options::String help{
        "The extraction radius of the spherical solution"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct Mass {
    using type = double;
    static constexpr Options::String help{
        "The mass of the Schwarzschild black hole"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct Frequency {
    using type = double;
    static constexpr Options::String help{
        "The frequency of the coordinate rotation."};
    static type lower_bound() noexcept { return 0.0; }
  };

  using options = tmpl::list<ExtractionRadius, Mass, Frequency>;

  static constexpr Options::String help = {
      "Analytic solution representing a Schwarzschild black hole in a rotating "
      "frame"};

  WRAPPED_PUPable_decl_template(RotatingSchwarzschild);  // NOLINT

  explicit RotatingSchwarzschild(CkMigrateMessage* /*unused*/) noexcept {}

  // clang doesn't manage to use = default correctly in this case
  // NOLINTNEXTLINE(modernize-use-equals-default)
  RotatingSchwarzschild() noexcept {};

  RotatingSchwarzschild(double extraction_radius, double mass,
                        double frequency) noexcept;

  std::unique_ptr<WorldtubeData> get_clone() const noexcept override;

  void pup(PUP::er& p) noexcept override;

 protected:
  /// A no-op as the rotating Schwarzschild solution does not have substantial
  /// shared computation to prepare before the separate component calculations.
  void prepare_solution(const size_t /*output_l_max*/,
                        const double /*time*/) const noexcept override {}

  /*!
   * \brief Compute the spherical coordinate metric from the closed-form
   * rotating Schwarzschild metric.
   *
   * \details The rotating Schwarzschild takes the coordinate form
   * \cite Barkett2019uae,
   *
   * \f{align}{
   * ds^2 = -\left(1 - \frac{2 M}{r} - \omega^2 r^2 \sin^2 \theta\right) dt^2
   * + \frac{1}{1 - \frac{2 M}{r}} dr^2
   * + 2 \omega r^2 \sin^2 \theta dt d\phi + r^2 d\Omega^2
   * \f}
   */
  void spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          spherical_metric,
      size_t l_max, double time) const noexcept override;

  /*!
   * \brief Compute the radial derivative of the spherical coordinate metric
   * from the closed-form rotating Schwarzschild metric.
   *
   * \details The rotating Schwarzschild takes the coordinate form
   * \cite Barkett2019uae,
   *
   * \f{align}{
   * \partial_r g_{a b} dx^a dx^b =& -\left(\frac{2 M}{r^2} - 2 \omega^2 r
   * \sin^2 \theta\right) dt^2 - \frac{2 M}{(r - 2 M)^2} dr^2
   * + 4 \omega r \sin^2 \theta dt d\phi + 2 r d\Omega^2
   * \f}
   */
  void dr_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dr_spherical_metric,
      size_t l_max, double time) const noexcept override;

  /*!
   * \brief The time derivative of the spherical coordinate metric in the
   * rotating Schwarzschild metric vanishes.
   */
  void dt_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dt_spherical_metric,
      size_t l_max, double time) const noexcept override;

  using WorldtubeData::variables_impl;

  using SphericalMetricData::variables_impl;

  /// The News vanishes in the rotating Schwarzschild metric
  void variables_impl(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
      size_t l_max, double time,
      tmpl::type_<Tags::News> /*meta*/) const noexcept override;

  double frequency_ = std::numeric_limits<double>::signaling_NaN();
  double mass_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Cce::Solutions
