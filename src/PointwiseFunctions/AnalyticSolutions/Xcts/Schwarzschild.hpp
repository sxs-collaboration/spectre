// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
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

}  // namespace Xcts::Solutions

template <>
struct Options::create_from_yaml<Xcts::Solutions::SchwarzschildCoordinates> {
  template <typename Metavariables>
  static Xcts::Solutions::SchwarzschildCoordinates create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
Xcts::Solutions::SchwarzschildCoordinates
Options::create_from_yaml<Xcts::Solutions::SchwarzschildCoordinates>::create<
    void>(const Options::Option& options);

namespace Xcts::Solutions {

namespace detail {

struct SchwarzschildImpl {
  struct Mass {
    using type = double;
    static constexpr Options::String help = "Mass parameter M";
  };

  struct CoordinateSystem {
    static std::string name() noexcept { return "Coordinates"; }
    using type = SchwarzschildCoordinates;
    static constexpr Options::String help =
        "The coordinate system used to describe the solution";
  };

  using options = tmpl::list<Mass, CoordinateSystem>;
  static constexpr Options::String help{
      "Schwarzschild spacetime in general relativity"};

  SchwarzschildImpl() = default;
  SchwarzschildImpl(const SchwarzschildImpl&) noexcept = default;
  SchwarzschildImpl& operator=(const SchwarzschildImpl&) noexcept = default;
  SchwarzschildImpl(SchwarzschildImpl&&) noexcept = default;
  SchwarzschildImpl& operator=(SchwarzschildImpl&&) noexcept = default;
  ~SchwarzschildImpl() noexcept = default;

  explicit SchwarzschildImpl(
      double mass, SchwarzschildCoordinates coordinate_system) noexcept;

  /// The mass parameter \f$M\f$.
  double mass() const noexcept;

  SchwarzschildCoordinates coordinate_system() const noexcept;

  /// The radius of the Schwarzschild horizon in the given coordinates.
  double radius_at_horizon() const noexcept;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 protected:
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  SchwarzschildCoordinates coordinate_system_{};
};

bool operator==(const SchwarzschildImpl& lhs,
                const SchwarzschildImpl& rhs) noexcept;

bool operator!=(const SchwarzschildImpl& lhs,
                const SchwarzschildImpl& rhs) noexcept;

template <typename DataType>
struct SchwarzschildVariables {
  using Cache = CachedTempBuffer<
      SchwarzschildVariables,
      Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataType>,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                    tmpl::size_t<3>, Frame::Inertial>,
      Tags::ConformalFactor<DataType>,
      ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial>,
      Tags::LapseTimesConformalFactor<DataType>,
      ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial>,
      Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
      Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
      Tags::ShiftStrain<DataType, 3, Frame::Inertial>,
      gr::Tags::EnergyDensity<DataType>, gr::Tags::StressTrace<DataType>,
      gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>,
      ::Tags::FixedSource<Tags::ConformalFactor<DataType>>,
      ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>>,
      ::Tags::FixedSource<Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;

  const tnsr::I<DataType, 3>& x;
  double mass;
  SchwarzschildCoordinates coordinate_system;

  void operator()(gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> trace_extrinsic_curvature_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalFactor<DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> conformal_factor_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      Tags::LapseTimesConformalFactor<DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*>
          lapse_times_conformal_factor_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
                  gsl::not_null<Cache*> cache,
                  Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> shift_strain,
      gsl::not_null<Cache*> cache,
      Tags::ShiftStrain<DataType, 3, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> energy_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::EnergyDensity<DataType> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> stress_trace,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::StressTrace<DataType> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> momentum_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::MomentumDensity<3, Frame::Inertial,
                                            DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> fixed_source_for_hamiltonian_constraint,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<Tags::ConformalFactor<DataType>> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> fixed_source_for_lapse_equation,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> fixed_source_momentum_constraint,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<
          Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
      const noexcept;
};

}  // namespace detail

// The following implements the registration and factory-creation mechanism

/// \cond
template <typename Registrars>
struct Schwarzschild;

namespace Registrars {
struct Schwarzschild {
  template <typename Registrars>
  using f = Solutions::Schwarzschild<Registrars>;
};
}  // namespace Registrars
/// \endcond

/*!
 * \brief Schwarzschild spacetime in general relativity
 *
 * This class implements the Schwarzschild solution with mass parameter
 * \f$M\f$ in various coordinate systems. See the entries of the
 * `Xcts::Solutions::SchwarzschildCoordinates` enum for the available coordinate
 * systems and for the solution variables in the respective coordinates.
 */
template <typename Registrars =
              tmpl::list<Solutions::Registrars::Schwarzschild>>
class Schwarzschild : public AnalyticSolution<Registrars>,
                      public detail::SchwarzschildImpl {
 private:
  using Base = AnalyticSolution<Registrars>;

 public:
  Schwarzschild() = default;
  Schwarzschild(const Schwarzschild&) noexcept = default;
  Schwarzschild& operator=(const Schwarzschild&) noexcept = default;
  Schwarzschild(Schwarzschild&&) noexcept = default;
  Schwarzschild& operator=(Schwarzschild&&) noexcept = default;
  ~Schwarzschild() noexcept = default;

  using SchwarzschildImpl::SchwarzschildImpl;

  /// \cond
  explicit Schwarzschild(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Schwarzschild);
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    using VarsComputer = detail::SchwarzschildVariables<DataType>;
    typename VarsComputer::Cache cache{
        get_size(*x.begin()), VarsComputer{x, mass_, coordinate_system_}};
    return {cache.get_var(RequestedTags{})...};
  }

  void pup(PUP::er& p) noexcept override {
    Base::pup(p);
    detail::SchwarzschildImpl::pup(p);
  }
};

/// \cond
template <typename Registrars>
PUP::able::PUP_ID Schwarzschild<Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Xcts::Solutions
