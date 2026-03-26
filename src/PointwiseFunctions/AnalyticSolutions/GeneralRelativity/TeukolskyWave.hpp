// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <string>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

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
 * \brief A perturbative Teukolsky wave (optionally plus Minkowski).
 *
 * \details This class provides a linearized \f$l=2\f$ Teukolsky wave in
 * Cartesian inertial coordinates. The implementation evaluates the spherical
 * formulas point-by-point and then transforms to Cartesian coordinates.
 *
 * This solution does not provide spatial derivatives of the metric, so it
 * does not support the full tag list of a typical GR analytic solution.
 *
 * See Eqs. (5) -- (10) of \cite Teukolsky1982nz for the equations in
 * spherical coordinates of the perturbed metric implemented here. Those
 * equations contain a freely specifiable radial profile; this implementation
 * chooses a Gaussian in \f$r - R_0 \pm t\f$, where \f$R_0\f$ is the
 * `Radius` parameter, with `Width` setting the pulse width.
 *
 * \note The implementation evaluates spherical-coordinate expressions, so it
 * must not be used too close to the origin about `Center`, where those
 * coordinates are singular. Specifically, evaluating the solution at points
 * with \f$r \le 0.1\f$, where \f$r\f$ is the distance from `Center`,
 * triggers an error. This cutoff is unrelated to the option `Radius`,
 * which sets the location of the Gaussian pulse.
 *
 * On the \f$z\f$-axis, the spherical coordinate \f$\phi\f$ is not well
 * defined. The implementation here is based on an implementation in SpEC that
 * just assumed the solution is never evaluated on the axis. Here, the
 * z-axis singularity is handled explicitly: on the \f$z\f$-axis
 * (cylindrical radius \f$\rho = 0\f$), the Cartesian result is obtained by
 * averaging the smooth limit approached from two orthogonal transverse
 * directions, so the solution is regular there.
 *
 * Optionally, the solution can include a flat Minkowski background as well as
 * the metric perturbation. Note that the tags for the inverse spatial metric,
 * square root of the spatial metric determinant, and extrinsic curvature
 * require the flat background to be included. The background can be excluded
 * when one wants only the perturbation itself (e.g., to add it on top of a
 * different background, such as the metric of a Kerr-Schild black hole).
 */
class TeukolskyWave : public MarkAsAnalyticSolution {
 public:
  static constexpr size_t volume_dim = 3;

  struct Amplitude {
    using type = double;
    static constexpr Options::String help{"Amplitude of the perturbation"};
  };

  struct Mode {
    using type = int;
    static constexpr Options::String help{
        "Azimuthal mode m of the l=2 Teukolsky wave"};
    static type lower_bound() { return -2; }
    static type upper_bound() { return 2; }
  };

  struct Parity {
    using type = std::string;
    static constexpr Options::String help{
        "Parity of the perturbation: 'even' or 'odd'"};
  };

  struct Direction {
    using type = std::string;
    static constexpr Options::String help{
        "Propagation direction: 'outgoing' or 'ingoing'"};
  };

  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help{
        "Center of the Teukolsky wave in inertial coordinates"};
  };

  struct Radius {
    using type = double;
    static constexpr Options::String help{
        "Radius of the center of the Gaussian pulse at t=0"};
  };

  struct Width {
    using type = double;
    static constexpr Options::String help{
        "Width of the Gaussian pulse profile"};
    static type lower_bound() { return 0.0; }
  };

  using options =
      tmpl::list<Amplitude, Mode, Parity, Direction, Center, Radius, Width>;
  static constexpr Options::String help{
      "A perturbative Teukolsky wave in Cartesian inertial coordinates"};

  template <typename DataType>
  using tags = tmpl::list<
      gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
      gr::Tags::Shift<DataType, volume_dim, Frame::Inertial>,
      ::Tags::dt<gr::Tags::Shift<DataType, volume_dim, Frame::Inertial>>,
      gr::Tags::SpatialMetric<DataType, volume_dim, Frame::Inertial>,
      ::Tags::dt<
          gr::Tags::SpatialMetric<DataType, volume_dim, Frame::Inertial>>,
      gr::Tags::SqrtDetSpatialMetric<DataType>,
      gr::Tags::ExtrinsicCurvature<DataType, volume_dim, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<DataType, volume_dim, Frame::Inertial>>;

  TeukolskyWave(double amplitude, int mode, std::string parity,
                std::string direction, std::array<double, 3> center,
                double radius, double width,
                const Options::Context& context = {});

  TeukolskyWave(double amplitude, int mode, std::string parity,
                std::string direction, std::array<double, 3> center,
                double radius, double width, bool include_minkowski_background,
                const Options::Context& context = {});

  TeukolskyWave() = default;
  TeukolskyWave(const TeukolskyWave& /*rhs*/) = default;
  TeukolskyWave& operator=(const TeukolskyWave& /*rhs*/) = default;
  TeukolskyWave(TeukolskyWave&& /*rhs*/) = default;
  TeukolskyWave& operator=(TeukolskyWave&& /*rhs*/) = default;
  ~TeukolskyWave() = default;

  explicit TeukolskyWave(CkMigrateMessage* /*msg*/);

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<RequestedTags...> /*meta*/) const {
    static_assert(tmpl2::flat_all_v<
                      tmpl::list_contains_v<tags<DataType>, RequestedTags>...>,
                  "At least one of the requested tags is not supported.");
    return {
        get<RequestedTags>(variables(x, t, tmpl::list<RequestedTags>{}))...};
  }

  template <typename RequestedTag, typename DataType>
  tuples::TaggedTuple<RequestedTag> variable(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
      const double t) const {
    return variables(x, t, tmpl::list<RequestedTag>{});
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  double amplitude() const { return amplitude_; }
  int mode() const { return mode_; }
  const std::string& parity() const { return parity_; }
  const std::string& direction() const { return direction_; }
  const std::array<double, 3>& center() const { return center_; }
  double radius() const { return radius_; }
  double width() const { return width_; }
  bool include_minkowski_background() const {
    return include_minkowski_background_;
  }

 private:
  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const
      -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<gr::Tags::Shift<DataType, volume_dim, Frame::Inertial>>
      /*meta*/) const
      -> tuples::TaggedTuple<
          gr::Tags::Shift<DataType, volume_dim, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t,
                 tmpl::list<::Tags::dt<gr::Tags::Shift<
                     DataType, volume_dim, Frame::Inertial>>> /*meta*/) const
      -> tuples::TaggedTuple<
          ::Tags::dt<gr::Tags::Shift<DataType, volume_dim, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t,
                 tmpl::list<gr::Tags::SpatialMetric<
                     DataType, volume_dim, Frame::Inertial>> /*meta*/) const
      -> tuples::TaggedTuple<
          gr::Tags::SpatialMetric<DataType, volume_dim, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<::Tags::dt<
          gr::Tags::SpatialMetric<DataType, volume_dim, Frame::Inertial>>>
      /*meta*/) const
      -> tuples::TaggedTuple<::Tags::dt<
          gr::Tags::SpatialMetric<DataType, volume_dim, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<
          gr::Tags::ExtrinsicCurvature<DataType, volume_dim, Frame::Inertial>>
      /*meta*/) const
      -> tuples::TaggedTuple<
          gr::Tags::ExtrinsicCurvature<DataType, volume_dim, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t,
                 tmpl::list<gr::Tags::InverseSpatialMetric<
                     DataType, volume_dim, Frame::Inertial>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<
          DataType, volume_dim, Frame::Inertial>>;

  double amplitude_{std::numeric_limits<double>::signaling_NaN()};
  int mode_{0};
  std::string parity_{};
  std::string direction_{};
  std::array<double, 3> center_{};
  double radius_{std::numeric_limits<double>::signaling_NaN()};
  double width_{std::numeric_limits<double>::signaling_NaN()};
  bool include_minkowski_background_{false};
};

bool operator==(const TeukolskyWave& lhs, const TeukolskyWave& rhs);
bool operator!=(const TeukolskyWave& lhs, const TeukolskyWave& rhs);

}  // namespace gr::Solutions
