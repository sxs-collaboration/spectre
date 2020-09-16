// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"
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

namespace gr {
namespace Solutions {

/*!
 * \brief Gauge wave in flat spacetime
 *
 * \details
 * This solution is Minkowski space in coordinates chosen to contain
 * a gauge wave. The spacetime metric is given by Eq. (4.3) of
 * \cite Alcubierre2003pc :
 *
 * \f{equation}{
 * ds^2 = -H dt^2 + H dx^2 + dy^2 + dz^2,
 * \f}
 *
 * where
 *
 * \f{equation}{
 * H = H(x-t) = 1 - A \sin\left(\frac{2\pi(x-t)}{d}\right).
 * \f}
 *
 * The gauge wave has amplitude \f$A\f$, wavelength \f$d\f$, and propagates
 * along the x axis.
 *
 * In these coordinates, the spatial metric \f$\gamma_{ij}\f$ and
 * inverse spatial metric \f$\gamma^{ij}\f$ are diagonal,
 * with the diagonal elements equal to unity except for
 *
 * \f{align}{
 * \gamma_{xx} & = H,\\
 * \gamma^{xx} & = 1/H.
 * \f}
 *
 * The components of the derivatives of \f$\gamma_{ij}\f$ vanish except for
 *
 * \f{align}{
 * \partial_t \gamma_{xx} & = \partial_t H = - \partial_x H,\\
 * \partial_x \gamma_{xx} & = \partial_x H.
 * \f}
 *
 * The square root of the spatial metric determinant is
 *
 * \f{align}{
 * \sqrt{\gamma} & = \sqrt{H}.
 * \f}
 *
 * The lapse and its derivatives are
 *
 * \f{align}{
 * \alpha & = \sqrt{H},\\
 * \partial_t \alpha & = -\frac{\partial_x H}{2\sqrt{H}},\\
 * \partial_x \alpha & = \frac{\partial_x H}{2\sqrt{H}},\\
 * \partial_y \alpha & = \partial_z \alpha = 0.
 * \f}
 *
 * The shift \f$\beta^i\f$ and its derivatives vanish.
 *
 * The extrinsic curvature's components vanish, except for
 *
 * \f{align}{
 * K_{xx} & = \frac{\partial_x H}{2 \sqrt{H}}.
 * \f}
 *
 * The following are input file options that can be specified:
 *  - Amplitude
 *  - Wavelength
 */
template <size_t Dim>
class GaugeWave : public MarkAsAnalyticSolution {
  template <typename DataType>
  struct IntermediateVars;

 public:
  static constexpr size_t volume_dim = Dim;
  struct Amplitude {
    using type = double;
    static constexpr Options::String help = {"Amplitude of the gauge wave"};
    static type upper_bound() noexcept { return 1.; }
    static type lower_bound() noexcept { return -1.; }
  };
  struct Wavelength {
    using type = double;
    static constexpr Options::String help = {"Wavelength of the gauge wave"};
    static type lower_bound() noexcept { return 0.; }
  };

  using options = tmpl::list<Amplitude, Wavelength>;
  static constexpr Options::String help{"Gauge wave in flat spacetime"};

  GaugeWave(double amplitude, double wavelength,
            const Options::Context& context = {});

  GaugeWave() = default;
  GaugeWave(const GaugeWave& /*rhs*/) = default;
  GaugeWave& operator=(const GaugeWave& /*rhs*/) = default;
  GaugeWave(GaugeWave&& /*rhs*/) noexcept = default;
  GaugeWave& operator=(GaugeWave&& /*rhs*/) noexcept = default;
  ~GaugeWave() = default;

  template <typename DataType>
  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataType>,
                                   tmpl::size_t<volume_dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivShift =
      ::Tags::deriv<gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>,
                    tmpl::size_t<volume_dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivSpatialMetric = ::Tags::deriv<
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>,
      tmpl::size_t<volume_dim>, Frame::Inertial>;

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    const auto& vars =
        IntermediateVars<DataType>{amplitude_, wavelength_, x, t};
    return {get<Tags>(variables(x, t, vars, tmpl::list<Tags>{}))...};
  }

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      const IntermediateVars<DataType>& vars,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "Unrecognized tag requested.  See the function parameters "
                  "for the tag.");
    return {get<Tags>(variables(x, t, vars, tmpl::list<Tags>{}))...};
  }

  template <typename DataType>
  using tags = tmpl::list<
      gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
      DerivLapse<DataType>,
      gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>,
      ::Tags::dt<gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>>,
      DerivShift<DataType>,
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>,
      ::Tags::dt<
          gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>>,
      DerivSpatialMetric<DataType>, gr::Tags::SqrtDetSpatialMetric<DataType>,
      gr::Tags::ExtrinsicCurvature<volume_dim, Frame::Inertial, DataType>,
      gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial, DataType>>;

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  SPECTRE_ALWAYS_INLINE double amplitude() const noexcept { return amplitude_; }
  SPECTRE_ALWAYS_INLINE double wavelength() const noexcept {
    return wavelength_;
  }

 private:
  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      const IntermediateVars<DataType>& vars,
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<DerivLapse<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<DerivLapse<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<gr::Tags::Shift<volume_dim, Frame::Inertial,
                                            DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<::Tags::dt<gr::Tags::Shift<
                     volume_dim, Frame::Inertial, DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::dt<gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<DerivShift<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<DerivShift<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<gr::Tags::SpatialMetric<volume_dim, Frame::Inertial,
                                                    DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<
                     volume_dim, Frame::Inertial, DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Tags::dt<
          gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
                 double /*t*/, const IntermediateVars<DataType>& vars,
                 tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<DerivSpatialMetric<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/, double /*t*/,
      const IntermediateVars<DataType>& vars,
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<gr::Tags::ExtrinsicCurvature<
                     volume_dim, Frame::Inertial, DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          gr::Tags::ExtrinsicCurvature<volume_dim, Frame::Inertial, DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                 double t, const IntermediateVars<DataType>& vars,
                 tmpl::list<gr::Tags::InverseSpatialMetric<
                     volume_dim, Frame::Inertial, DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<
          volume_dim, Frame::Inertial, DataType>>;

  template <typename DataType>
  struct IntermediateVars {
    IntermediateVars(double amplitude, double wavelength,
                     const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
                     double t) noexcept;
    DataType h{};
    DataType dx_h{};
    DataType sqrt_h{};
    DataType dx_h_over_2_sqrt_h{};
  };

  double amplitude_{1.0};
  double wavelength_{1.0};
};

template <size_t Dim>
bool operator==(const GaugeWave<Dim>& lhs, const GaugeWave<Dim>& rhs) noexcept;
template <size_t Dim>
bool operator!=(const GaugeWave<Dim>& lhs, const GaugeWave<Dim>& rhs) noexcept;
}  // namespace Solutions
}  // namespace gr
