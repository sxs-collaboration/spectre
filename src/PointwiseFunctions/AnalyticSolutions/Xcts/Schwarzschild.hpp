// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::Solutions {

/// Various coordinate systems in which to express the Schwarzschild solution
enum class SchwarzschildCoordinates {
  /*!
   * \brief Isotropic Schwarzschild coordinates
   *
   * These arise from the canonical Schwarzschild coordinates by the radial
   * transformation
   *
   * \f{equation}
   * r = \bar{r}\left(1+\frac{M}{2\bar{r}}\right)^2
   * \f}
   *
   * (Eq. (1.61) in \cite BaumgarteShapiro) where \f$r\f$ is the canonical
   * Schwarzschild radius, also referred to as "areal" radius because it is
   * defined such that spheres with constant \f$r\f$ have the area \f$4\pi
   * r^2\f$, and \f$\bar{r}\f$ is the "isotropic" radius. In the isotropic
   * radius the Schwarzschild spatial metric is conformally flat:
   *
   * \f{equation}
   * \gamma_{ij}=\psi^4\eta_{ij} \quad \text{with conformal factor} \quad
   * \psi=1+\frac{M}{2\bar{r}}
   * \f}
   *
   * (Table 2.1 in \cite BaumgarteShapiro). Its lapse transforms to
   *
   * \f{equation}
   * \alpha=\frac{1-M/(2\bar{r})}{1+M/(2\bar{r})}
   * \f}
   *
   * and the shift vanishes (\f$\beta^i=0\f$) as it does in areal Schwarzschild
   * coordinates. The solution also remains maximally sliced, i.e. \f$K=0\f$.
   *
   * The Schwarzschild horizon in these coordinates is at
   * \f$\bar{r}=\frac{M}{2}\f$ due to the radial transformation from \f$r=2M\f$.
   */
  Isotropic,
};

std::ostream& operator<<(std::ostream& os,
                         SchwarzschildCoordinates coords) noexcept;

/*!
 * \brief Schwarzschild spacetime in general relativity
 *
 * This class implements the Schwarzschild solution with mass parameter
 * \f$M\f$ in various coordinate systems. See the entries of the
 * `Xcts::Solutions::SchwarzschildCoordinates` enum for the available coordinate
 * systems and for the solution variables in the respective coordinates.
 */
template <SchwarzschildCoordinates Coords>
class Schwarzschild {
 private:
  struct Mass {
    using type = double;
    static constexpr Options::String help = "Mass parameter M";
  };

 public:
  using options = tmpl::list<Mass>;
  static constexpr Options::String help{
      "Schwarzschild spacetime in general relativity"};

  Schwarzschild() = default;
  Schwarzschild(const Schwarzschild&) noexcept = delete;
  Schwarzschild& operator=(const Schwarzschild&) noexcept = delete;
  Schwarzschild(Schwarzschild&&) noexcept = default;
  Schwarzschild& operator=(Schwarzschild&&) noexcept = default;
  ~Schwarzschild() noexcept = default;

  explicit Schwarzschild(double mass) noexcept;

  /// The mass parameter \f$M\f$.
  double mass() const noexcept;

  /// The radius of the Schwarzschild horizon in the given coordinates.
  double radius_at_horizon() const noexcept;

  // @{
  /// Retrieve variable at coordinates `x`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ConformalMetric<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                        Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ShiftBackground<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<
          Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::ConformalFactor<DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::FixedSource<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::StressTrace<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial,
                                                      DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1, "The requested tag is not implemented.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  void pup(PUP::er& p) noexcept {  // NOLINT
    p | mass_;
  }

 private:
  double mass_;
};

template <SchwarzschildCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator==(
    const Schwarzschild<Coords>& lhs,
    const Schwarzschild<Coords>& rhs) noexcept {
  return lhs.mass() == rhs.mass();
}

template <SchwarzschildCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const Schwarzschild<Coords>& lhs,
    const Schwarzschild<Coords>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Xcts::Solutions
