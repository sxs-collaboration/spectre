// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <ostream>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/ContainerHelpers.hpp"
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
   * (Table 2.1 in \cite BaumgarteShapiro). The lapse in the conformal radius is
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
  /*!
   * \brief Painlevé-Gullstrand coordinates
   *
   * In these coordinates the spatial metric is flat and the lapse is trivial,
   * but contrary to (isotropic) Schwarzschild coordinates the shift is
   * nontrivial,
   *
   * \begin{align}
   *   \gamma_{ij} &= \eta_{ij} \\
   *   \alpha &= 1 \\
   *   \beta^i &= \sqrt{\frac{2M}{r}} \frac{x^i}{r} \\
   *   K &= \frac{3}{2}\sqrt{\frac{2M}{r^3}}
   * \end{align}
   *
   * (Table 2.1 in \cite BaumgarteShapiro).
   */
  PainleveGullstrand,
  /*!
   * \brief Isotropic Kerr-Schild coordinates
   *
   * Kerr-Schild coordinates with a radial transformation such that the spatial
   * metric is conformally flat.
   *
   * The Schwarzschild spacetime in canonical (areal) Kerr-Schild coordinates is
   *
   * \begin{align}
   *   \gamma_{ij} &= \eta_{ij} + \frac{2M}{r}\frac{x^i x^j}{r^2} \\
   *   \alpha &= \sqrt{1 + \frac{2M}{r}}^{-1} \\
   *   \beta^i &= \frac{2M\alpha^2}{r} \frac{x^i}{r} \\
   *   K &= \frac{2M\alpha^3}{r^2} \left(1 + \frac{3M}{r}\right)
   *   \text{.}
   * \end{align}
   *
   * (Table 2.1 in \cite BaumgarteShapiro). Since the Schwarzschild spacetime is
   * spherically symmetric we can transform to a radial coordinate $\bar{r}$ in
   * which it is conformally flat (see, e.g., Sec. 7.4.1 in \cite Pfeiffer2005zm
   * for details):
   *
   * \begin{equation}
   *   {}^{(3)}\mathrm{d}s^2
   *     = \left(1 + \frac{2M}{r}\right)\mathrm{d}r^2 + r^2 \mathrm{d}\Omega^2
   *     = \psi^4 \left(\mathrm{d}\bar{r}^2 +
   *       \bar{r}^2 \mathrm{d}\Omega^2\right)
   * \end{equation}
   *
   * Therefore, the conformal factor is $\psi^2 = r / \bar{r}$ and
   *
   * \begin{equation}
   *   \frac{\mathrm{d}\bar{r}}{\mathrm{d}r}
   *     = \frac{\bar{r}}{r} \sqrt{1 + \frac{2M}{r}}
   *     = \frac{\bar{r}}{r} \frac{1}{\alpha}
   *   \text{,}
   * \end{equation}
   *
   * which has the solution
   *
   * \begin{equation}
   *   \bar{r} = \frac{r}{4} \left(1 + \sqrt{1 + \frac{2M}{r}}\right)^2
   *     e^{2 - 2\sqrt{1 + 2M / r}}
   * \end{equation}
   *
   * when we impose $\bar{r} \rightarrow r$ as $r \rightarrow \infty$. We can
   * invert this transformation law with a numerical root find to obtain the
   * areal radius $r$ for any isotropic radius $\bar{r}$.
   *
   * In the isotropic radial coordinate $\bar{r}$ the solution is then:
   *
   * \begin{align}
   *   \gamma_{ij} &= \psi^4 \eta_{ij} \\
   *   \psi &= \sqrt{\frac{r}{\bar{r}}}
   *     = \frac{2e^{\sqrt{1 + 2M / r} - 1}}{1 + \sqrt{1 + 2M / r}} \\
   *   \alpha &= \sqrt{1 + \frac{2M}{r}}^{-1} \\
   *   \beta^i
   *     &= \frac{\mathrm{d}\bar{r}}{\mathrm{d}r} \beta^r \frac{x^i}{\bar{r}}
   *      = \frac{2M\alpha}{r^2} x^i \\
   *   K &= \frac{2M\alpha^3}{r^2} \left(1 + \frac{3M}{r}\right)
   * \end{align}
   *
   * Here, $x^i$ are the (isotropic) Cartesian coordinates from which we compute
   * the isotropic radius $\bar{r}$, $r$ is the areal radius we can obtain from
   * the isotropic radius by a root find, and $\beta^r$ is the magnitude of the
   * shift in areal coordinates, as given above.
   */
  KerrSchildIsotropic,
};

std::ostream& operator<<(std::ostream& os, SchwarzschildCoordinates coords);

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
    static std::string name() { return "Coordinates"; }
    using type = SchwarzschildCoordinates;
    static constexpr Options::String help =
        "The coordinate system used to describe the solution";
  };

  using options = tmpl::list<Mass, CoordinateSystem>;
  static constexpr Options::String help{
      "Schwarzschild spacetime in general relativity"};

  SchwarzschildImpl() = default;
  SchwarzschildImpl(const SchwarzschildImpl&) = default;
  SchwarzschildImpl& operator=(const SchwarzschildImpl&) = default;
  SchwarzschildImpl(SchwarzschildImpl&&) = default;
  SchwarzschildImpl& operator=(SchwarzschildImpl&&) = default;
  ~SchwarzschildImpl() = default;

  explicit SchwarzschildImpl(double mass,
                             SchwarzschildCoordinates coordinate_system);

  /// The mass parameter \f$M\f$.
  double mass() const;

  SchwarzschildCoordinates coordinate_system() const;

  /// The radius of the Schwarzschild horizon in the given coordinates.
  double radius_at_horizon() const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 protected:
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  SchwarzschildCoordinates coordinate_system_{};
};

bool operator==(const SchwarzschildImpl& lhs, const SchwarzschildImpl& rhs);

bool operator!=(const SchwarzschildImpl& lhs, const SchwarzschildImpl& rhs);

namespace Tags {
template <typename DataType>
struct Radius : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct ArealRadius : db::SimpleTag {
  using type = Scalar<DataType>;
};
}  // namespace Tags

template <typename DataType>
using SchwarzschildVariablesCache =
    cached_temp_buffer_from_typelist<tmpl::push_front<
        common_tags<DataType>, detail::Tags::Radius<DataType>,
        detail::Tags::ArealRadius<DataType>,
        gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
        gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
        gr::Tags::Conformal<
            gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>, 0>>>;

template <typename DataType>
struct SchwarzschildVariables
    : CommonVariables<DataType, SchwarzschildVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  static constexpr int ConformalMatterScale = 0;
  using Cache = SchwarzschildVariablesCache<DataType>;
  using Base = CommonVariables<DataType, SchwarzschildVariablesCache<DataType>>;
  using Base::operator();

  SchwarzschildVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, 3>& local_x, const double local_mass,
      const SchwarzschildCoordinates local_coordinate_system)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        mass(local_mass),
        coordinate_system(local_coordinate_system) {}

  const tnsr::I<DataType, 3>& x;
  double mass;
  SchwarzschildCoordinates coordinate_system;

  void operator()(gsl::not_null<Scalar<DataType>*> radius,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::Radius<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> areal_radius,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::ArealRadius<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::II<DataType, 3>*> inv_conformal_metric,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::InverseConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> trace_extrinsic_curvature_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> conformal_factor,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ConformalFactor<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> conformal_factor_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const override;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::LapseTimesConformalFactor<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*>
          lapse_times_conformal_factor_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, 3, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> shift_strain,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<Scalar<DataType>*> energy_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>,
                                      ConformalMatterScale> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> stress_trace,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<gr::Tags::StressTrace<DataType>,
                                      ConformalMatterScale> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> momentum_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<
                      gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>,
                      ConformalMatterScale> /*meta*/) const;
};

}  // namespace detail

/*!
 * \brief Schwarzschild spacetime in general relativity
 *
 * This class implements the Schwarzschild solution with mass parameter
 * \f$M\f$ in various coordinate systems. See the entries of the
 * `Xcts::Solutions::SchwarzschildCoordinates` enum for the available coordinate
 * systems and for the solution variables in the respective coordinates.
 */
class Schwarzschild : public elliptic::analytic_data::AnalyticSolution,
                      public detail::SchwarzschildImpl {
 public:
  Schwarzschild() = default;
  Schwarzschild(const Schwarzschild&) = default;
  Schwarzschild& operator=(const Schwarzschild&) = default;
  Schwarzschild(Schwarzschild&&) = default;
  Schwarzschild& operator=(Schwarzschild&&) = default;
  ~Schwarzschild() = default;

  using SchwarzschildImpl::SchwarzschildImpl;

  /// \cond
  explicit Schwarzschild(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Schwarzschild);
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Schwarzschild>(*this);
  }
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::SchwarzschildVariables<DataType>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{std::nullopt, std::nullopt, x, mass_,
                                coordinate_system_};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::SchwarzschildVariables<DataVector>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{mesh, inv_jacobian, x, mass_,
                                coordinate_system_};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    detail::SchwarzschildImpl::pup(p);
  }
};

}  // namespace Xcts::Solutions
