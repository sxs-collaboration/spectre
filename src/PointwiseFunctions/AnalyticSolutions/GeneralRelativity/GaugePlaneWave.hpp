// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
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

namespace gr::Solutions {

/*!
 * \brief Gauge plane wave in flat spacetime
 *
 * \details
 * The spacetime metric is in Kerr-Schild form:
 * \f{equation}{
 * g_{\mu\nu} = \eta_{\mu\nu} + H l_\mu l_\nu
 * \f}
 * where \f$H\f$ is a scalar function of the coordinates, \f$\eta_{\mu\nu}\f$
 * is the Minkowski metric, and \f$l^\mu\f$ is a null vector. Note that the
 * form of the metric along with the nullness of \f$l^\mu\f$ allows one to
 * raise and lower indices of \f$l^\mu\f$ using \f$\eta_{\mu\nu}\f$, and
 * that \f$l^t l^t = l_t l_t = l^i l_i\f$.
 * Note also that
 * \f{equation}{
 * g^{\mu\nu}  \equiv \eta^{\mu\nu} - H l^\mu l^\nu,
 * \f}
 * and that \f$\sqrt{-g}=1\f$.
 * Also, \f$l_\mu\f$ is a geodesic with respect to both the physical metric
 * and the Minkowski metric:
 * \f{equation}{
 * l^\mu \partial_\mu l_\nu = l^\mu\nabla_\mu l_\nu = 0.
 * \f}
 *
 * For this solution we choose the profile \f$H\f$ of the plane wave to be an
 * arbitrary one-dimensional function of \f$u = \vec{k} \cdot \vec{x} - \omega
 * t\f$ with a constant Euclidean wave vector \f$\vec{k}\f$ and frequency
 * \f$\omega = ||\vec{k}||\f$.  The null covector is chosen to be \f$l_a =
 * (-\omega, k_i)\f$. Thus, if \f$H = H[u]\f$, then \f$\partial_\mu H = H'[u]
 * l_\mu\f$. Therefore the derivatives of the spacetime metric are:
 * \f{equation}{
 * \partial_\rho g_{\mu\nu} = H' l_\rho l_\mu l_\nu,
 * \f}
 *
 * The 3+1 quantities are
 * \f{align}{
 * \alpha & = \left( 1 + H \omega^2 \right)^{-1/2},\\
 * \beta^i & = \frac{-H \omega k^i}{1 + H \omega^2},\\
 * \gamma_{ij} & = \delta_{ij} + H k_i k_j,\\
 * \gamma & = 1 + H \omega^2,\\
 * \gamma^{ij} & = \delta^{ij} - \frac{H k^i k^j}{1 + H \omega^2},\\
 * \partial_t \alpha & = \frac{\omega^3 H'}{2 \left(1 + H \omega^2
 *    \right)^{3/2}},\\
 * \partial_i \alpha & = - \frac{\omega^2 H' k_i}{2 \left(1 + H
 *    \omega^2 \right)^{3/2}},\\
 * \partial_t \beta^i & = \frac{\omega^2 H' k^i}{\left(1 + H \omega^2
 *    \right)^2},\\
 * \partial_j \beta^i & = - \frac{\omega H' k_j k^i}{\left(1 + H
 *    \omega^2 \right)^2},\\
 * \partial_t \gamma_{ij} & = - \omega H' k_i k_j,\\
 * \partial_k \gamma_{ij} & = H' k_k k_i k_j,\\
 * K_{ij} & =  - \frac{\omega H' k_i k_j}{2 \left(1 + H
 *    \omega^2 \right)^{1/2}}.
 * \f}
 *
 * Note that this solution is a gauge wave as \f$\Gamma^a{}_{bc} = \frac{1}{2}
 * H' l^a l_b l_c\f$ and thus \f$R^a{}_{bcd} = 0\f$.
 *
 * \tparam Dim the spatial dimension of the solution
 */
template <size_t Dim>
class GaugePlaneWave : public AnalyticSolution<Dim>,
                       public MarkAsAnalyticSolution {
  template <typename DataType>
  struct IntermediateVars;

 public:
  static constexpr size_t volume_dim = Dim;
  struct WaveVector {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {
        "The direction of propagation of the wave."};
  };

  struct Profile {
    using type = std::unique_ptr<MathFunction<1, Frame::Inertial>>;
    static constexpr Options::String help = {"The profile of the wave."};
  };

  using options = tmpl::list<WaveVector, Profile>;
  static constexpr Options::String help{"Gauge plane wave in flat spacetime"};

  GaugePlaneWave() = default;
  GaugePlaneWave(std::array<double, Dim> wave_vector,
                 std::unique_ptr<MathFunction<1, Frame::Inertial>> profile);
  GaugePlaneWave(const GaugePlaneWave&);
  GaugePlaneWave& operator=(const GaugePlaneWave&);
  GaugePlaneWave(GaugePlaneWave&&) = default;
  GaugePlaneWave& operator=(GaugePlaneWave&&) = default;
  ~GaugePlaneWave() = default;

  explicit GaugePlaneWave(CkMigrateMessage* /*msg*/);

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

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, double t,
      tmpl::list<Tags...> /*meta*/) const {
    const auto& vars =
        IntermediateVars<DataType>{wave_vector_, profile_, omega_, x, t};
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
    IntermediateVars(
        const std::array<double, Dim>& wave_vector,
        const std::unique_ptr<MathFunction<1, Frame::Inertial>>& profile,
        double omega, const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
        double t);
    DataType h{};
    DataType du_h{};
    DataType det_gamma{};
    DataType lapse{};
  };

  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const GaugePlaneWave<LocalDim>& lhs,
                         const GaugePlaneWave<LocalDim>& rhs);
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator!=(const GaugePlaneWave<LocalDim>& lhs,
                         const GaugePlaneWave<LocalDim>& rhs);

  std::array<double, Dim> wave_vector_{};
  std::unique_ptr<MathFunction<1, Frame::Inertial>> profile_;
  double omega_{};
};
}  // namespace gr::Solutions
