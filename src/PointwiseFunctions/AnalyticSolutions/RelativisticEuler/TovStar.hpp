// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Solutions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Tov.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace RelativisticEuler::Solutions {
namespace tov_detail {

enum class StarRegion { Center, Interior, Exterior };

namespace Tags {
template <typename DataType>
struct MassOverRadius : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct LogSpecificEnthalpy : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct ConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct DrConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct ArealRadius : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct DrArealRadius : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct DrPressure : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct MetricTimePotential : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct DrMetricTimePotential : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct MetricRadialPotential : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct DrMetricRadialPotential : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct MetricAngularPotential : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct DrMetricAngularPotential : db::SimpleTag {
  using type = Scalar<DataType>;
};
}  // namespace Tags

template <typename DataType>
using TovVariablesCache = cached_temp_buffer_from_typelist<tmpl::list<
    Tags::MassOverRadius<DataType>, Tags::LogSpecificEnthalpy<DataType>,
    Tags::ConformalFactor<DataType>, Tags::DrConformalFactor<DataType>,
    Tags::ArealRadius<DataType>, Tags::DrArealRadius<DataType>,
    hydro::Tags::SpecificEnthalpy<DataType>,
    hydro::Tags::RestMassDensity<DataType>,
    hydro::Tags::ElectronFraction<DataType>, hydro::Tags::Pressure<DataType>,
    Tags::DrPressure<DataType>,
    ::Tags::deriv<hydro::Tags::Pressure<DataType>, tmpl::size_t<3>,
                  Frame::Inertial>,
    hydro::Tags::SpecificInternalEnergy<DataType>,
    Tags::MetricTimePotential<DataType>, Tags::DrMetricTimePotential<DataType>,
    Tags::MetricRadialPotential<DataType>,
    Tags::DrMetricRadialPotential<DataType>,
    Tags::MetricAngularPotential<DataType>,
    Tags::DrMetricAngularPotential<DataType>,
    hydro::Tags::LorentzFactor<DataType>,
    hydro::Tags::SpatialVelocity<DataType, 3>,
    hydro::Tags::MagneticField<DataType, 3>,
    hydro::Tags::DivergenceCleaningField<DataType>, gr::Tags::Lapse<DataType>,
    ::Tags::dt<gr::Tags::Lapse<DataType>>,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::Shift<DataType, 3>, ::Tags::dt<gr::Tags::Shift<DataType, 3>>,
    ::Tags::deriv<gr::Tags::Shift<DataType, 3>, tmpl::size_t<3>,
                  Frame::Inertial>,
    gr::Tags::SpatialMetric<DataType, 3>,
    ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3>>,
    ::Tags::deriv<gr::Tags::SpatialMetric<DataType, 3>, tmpl::size_t<3>,
                  Frame::Inertial>,
    gr::Tags::SqrtDetSpatialMetric<DataType>,
    gr::Tags::ExtrinsicCurvature<DataType, 3>,
    gr::Tags::InverseSpatialMetric<DataType, 3>>>;

template <typename DataType, StarRegion Region>
struct TovVariables {
  static constexpr size_t Dim = 3;
  using Cache = TovVariablesCache<DataType>;

  TovVariables(const TovVariables&) = default;
  TovVariables& operator=(const TovVariables&) = default;
  TovVariables(TovVariables&&) = default;
  TovVariables& operator=(TovVariables&&) = default;
  virtual ~TovVariables() = default;
  TovVariables(
      const tnsr::I<DataType, 3>& local_coords, const DataType& local_radius,
      const RelativisticEuler::Solutions::TovSolution& local_radial_solution,
      const EquationsOfState::EquationOfState<true, 1>& local_eos)
      : coords(local_coords),
        radius(local_radius),
        radial_solution(local_radial_solution),
        eos(local_eos) {}

  const tnsr::I<DataType, 3>& coords;
  const DataType& radius;
  const RelativisticEuler::Solutions::TovSolution& radial_solution;
  const EquationsOfState::EquationOfState<true, 1>& eos;

  void operator()(gsl::not_null<Scalar<DataType>*> mass_over_radius,
                  gsl::not_null<Cache*> cache,
                  Tags::MassOverRadius<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> log_specific_enthalpy,
                  gsl::not_null<Cache*> cache,
                  Tags::LogSpecificEnthalpy<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalFactor<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> dr_conformal_factor,
                  gsl::not_null<Cache*> cache,
                  Tags::DrConformalFactor<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> areal_radius,
                  gsl::not_null<Cache*> cache,
                  Tags::ArealRadius<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> dr_areal_radius,
                  gsl::not_null<Cache*> cache,
                  Tags::DrArealRadius<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> specific_enthalpy,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::SpecificEnthalpy<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> rest_mass_density,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::RestMassDensity<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> electron_fraction,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::ElectronFraction<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> pressure,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::Pressure<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> dr_pressure,
                  gsl::not_null<Cache*> cache,
                  Tags::DrPressure<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> deriv_pressure,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<hydro::Tags::Pressure<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> specific_internal_energy,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::SpecificInternalEnergy<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> metric_time_potential,
                  gsl::not_null<Cache*> cache,
                  Tags::MetricTimePotential<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> dr_metric_time_potential,
                  gsl::not_null<Cache*> cache,
                  Tags::DrMetricTimePotential<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> metric_radial_potential,
                  gsl::not_null<Cache*> cache,
                  Tags::MetricRadialPotential<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> dr_metric_radial_potential,
                  gsl::not_null<Cache*> cache,
                  Tags::DrMetricRadialPotential<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> metric_angular_potential,
                  gsl::not_null<Cache*> cache,
                  Tags::MetricAngularPotential<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> dr_metric_angular_potential,
                  gsl::not_null<Cache*> cache,
                  Tags::DrMetricAngularPotential<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> lorentz_factor,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::LorentzFactor<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::SpatialVelocity<DataType, 3> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
      gsl::not_null<Cache*> cache,
      hydro::Tags::MagneticField<DataType, 3> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> div_cleaning_field,
      gsl::not_null<Cache*> cache,
      hydro::Tags::DivergenceCleaningField<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> dt_lapse,
                  gsl::not_null<Cache*> cache,
                  ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::i<DataType, 3>*> deriv_lapse,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                                Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> shift,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Shift<DataType, 3> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> dt_shift,
                  gsl::not_null<Cache*> cache,
                  ::Tags::dt<gr::Tags::Shift<DataType, 3>> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::iJ<DataType, 3>*> deriv_shift,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<gr::Tags::Shift<DataType, 3>, tmpl::size_t<3>,
                                Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, 3>*> spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::SpatialMetric<DataType, 3> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> dt_spatial_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3>> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_spatial_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::SpatialMetric<DataType, 3>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> sqrt_det_spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, 3>*> extrinsic_curvature,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::ExtrinsicCurvature<DataType, 3> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::II<DataType, 3>*> inv_spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::InverseSpatialMetric<DataType, 3> /*meta*/) const;
};
}  // namespace tov_detail

/*!
 * \brief A static spherically symmetric star
 *
 * An analytic solution for a static, spherically-symmetric star found by
 * solving the Tolman-Oppenheimer-Volkoff (TOV) equations.  The equation of
 * state is assumed to be that of a polytropic fluid.
 *
 * If the spherically symmetric metric is written as
 *
 * \f[
 * ds^2 = - e^{2 \Phi_t} dt^2 + e^{2 \Phi_r} dr^2 + e^{2 \Phi_\Omega} r^2
 * d\Omega^2
 * \f]
 *
 * where \f$r = \delta_{mn} x^m x^n\f$ is the radial coordinate and
 * \f$\Phi_t\f$, \f$\Phi_r\f$, and \f$\Phi_\Omega\f$ are the metric potentials,
 * then the lapse, shift, and spatial metric in Cartesian coordinates are
 *
 * \f{align*}
 * \alpha &= e^{\Phi_t} \\
 * \beta^i &= 0 \\
 * \gamma_{ij} &= \delta_{ij} e^{2 \Phi_r} + \delta_{im} \delta_{jn}
 * \frac{x^m x^n}{r^2} \left( e^{2 \Phi_r} - e^{2 \Phi_\Omega} \right)
 * \f}
 *
 * We solve the TOV equations with the method implemented in
 * `RelativisticEuler::Solutions::TovSolution`. It provides the areal
 * mass-over-radius \f$m(r)/r\f$ and the log of the specific enthalpy
 * \f$\log{h}\f$. In areal (Schwarzschild) coordinates the spatial metric
 * potentials are
 *
 * \f{align}
 * e^{\Phi_r} &= \left(1 - \frac{2m}{r}\right)^{-1} \\
 * e^{\Phi_\Omega} &= 1
 * \f}
 *
 * In isotropic coordinates the spatial metric potentials are
 *
 * \begin{equation}
 * e^{2\Phi_r} = e^{2\Phi_\Omega} = \psi^4
 * \text{,}
 * \end{equation}
 *
 * where $\psi = \sqrt{r / \bar{r}}$ is the conformal factor, $r$ is the areal
 * (Schwarzschild) radius and $\bar{r}$ is the isotropic radius. See
 * `RelativisticEuler::Solutions::TovSolution` for details.
 */

class TovStar : public virtual evolution::initial_data::InitialData,
                public MarkAsAnalyticSolution,
                public AnalyticSolution<3> {
 public:
  using equation_of_state_type = EquationsOfState::EquationOfState<true, 1>;

  /// The central density of the star.
  struct CentralDensity {
    using type = double;
    static constexpr Options::String help = {
        "The central density of the star."};
    static type lower_bound() { return 0.; }
  };

  /// Areal (Schwarzschild) or isotropic coordinates
  struct Coordinates {
    using type = RelativisticEuler::Solutions::TovCoordinates;
    static constexpr Options::String help = {
        "Areal ('Schwarzschild') or 'Isotropic' coordinates."};
  };

  static constexpr size_t volume_dim = 3_st;

  using options =
      tmpl::list<CentralDensity, hydro::OptionTags::EquationOfState<true, 1>,
                 Coordinates>;

  static constexpr Options::String help = {
      "A static, spherically-symmetric star found by solving the \n"
      "Tolman-Oppenheimer-Volkoff (TOV) equations, with a given central \n"
      "density and equation of state."};

  TovStar() = default;
  TovStar(const TovStar& /*rhs*/);
  TovStar& operator=(const TovStar& /*rhs*/);
  TovStar(TovStar&& /*rhs*/) = default;
  TovStar& operator=(TovStar&& /*rhs*/) = default;
  ~TovStar() override = default;
  TovStar(double central_rest_mass_density,
          std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>
              equation_of_state,
          const RelativisticEuler::Solutions::TovCoordinates coordinate_system =
              RelativisticEuler::Solutions::TovCoordinates::Schwarzschild);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit TovStar(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TovStar);
  /// \endcond

  /// Retrieve a collection of variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         const double /*t*/,
                                         tmpl::list<Tags...> /*meta*/) const {
    return variables_impl<tov_detail::TovVariables>(x, tmpl::list<Tags...>{});
  }

  /// NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

  const EquationsOfState::EquationOfState<true, 1>& equation_of_state() const {
    return *equation_of_state_;
  }

  /// The radial profile of the star
  const RelativisticEuler::Solutions::TovSolution& radial_solution() const {
    return radial_solution_;
  }

 protected:
  template <template <class, tov_detail::StarRegion> class VarsComputer,
            typename DataType, typename... Tags, typename... VarsComputerArgs>
  tuples::TaggedTuple<Tags...> variables_impl(
      const tnsr::I<DataType, 3>& x, tmpl::list<Tags...> /*meta*/,
      VarsComputerArgs&&... vars_computer_args) const {
    const double outer_radius = radial_solution_.outer_radius();
    const double center_radius_cutoff = 1.e-30 * outer_radius;
    const DataType radius = get(magnitude(x));
    // Dispatch interior and exterior regions of the star.
    // - Include the equality in the conditions below for the outer radius so
    //   the `DataVector` variants are preferred over the pointwise `double`
    //   variant when possible.
    // - Order the conditions so the cheaper exterior solution is preferred over
    //   the interior.
    // - A `DataVector` variant for the center of the star is not needed because
    //   it's only a single point.
    if (min(radius) >= outer_radius) {
      // All points are outside the star. This could be replaced by a
      // Schwarzschild solution.
      using ExteriorVarsComputer =
          VarsComputer<DataType, tov_detail::StarRegion::Exterior>;
      typename ExteriorVarsComputer::Cache cache{get_size(radius)};
      ExteriorVarsComputer computer{
          x, radius, radial_solution_, *equation_of_state_,
          std::forward<VarsComputerArgs>(vars_computer_args)...};
      return {cache.get_var(computer, Tags{})...};
    } else if (max(radius) <= outer_radius and
               min(radius) > center_radius_cutoff) {
      // All points are in the star interior, but not at the center
      using InteriorVarsComputer =
          VarsComputer<DataType, tov_detail::StarRegion::Interior>;
      typename InteriorVarsComputer::Cache cache{get_size(radius)};
      InteriorVarsComputer computer{
          x, radius, radial_solution_, *equation_of_state_,
          std::forward<VarsComputerArgs>(vars_computer_args)...};
      return {cache.get_var(computer, Tags{})...};
    } else {
      // Points can be at the center, in the interior, or outside the star, so
      // check each point individually
      const size_t num_points = get_size(radius);
      tuples::TaggedTuple<Tags...> vars{typename Tags::type{num_points}...};
      const auto get_var = [&vars](const size_t point_index, auto& local_cache,
                                   const auto& local_computer, auto tag_v) {
        using tag = std::decay_t<decltype(tag_v)>;
        if constexpr (std::is_same_v<DataType, DataVector>) {
          using tags_double = typename VarsComputer<
              double, tov_detail::StarRegion::Exterior>::Cache::tags_list;
          using tags_dv = typename VarsComputer<
              DataVector, tov_detail::StarRegion::Exterior>::Cache::tags_list;
          using tag_double =
              tmpl::at<tags_double, tmpl::index_of<tags_dv, tag>>;
          const auto& tensor =
              local_cache.get_var(local_computer, tag_double{});
          for (size_t component = 0; component < tensor.size(); ++component) {
            get<tag>(vars)[component][point_index] = tensor[component];
          }
        } else {
          (void)point_index;
          get<tag>(vars) = local_cache.get_var(local_computer, tag{});
        }
        return '0';
      };
      using CenterVarsComputer =
          VarsComputer<double, tov_detail::StarRegion::Center>;
      using InteriorVarsComputer =
          VarsComputer<double, tov_detail::StarRegion::Interior>;
      using ExteriorVarsComputer =
          VarsComputer<double, tov_detail::StarRegion::Exterior>;
      for (size_t i = 0; i < num_points; ++i) {
        const tnsr::I<double, 3> x_i{
            {{get_element(get<0>(x), i), get_element(get<1>(x), i),
              get_element(get<2>(x), i)}}};
        if (get_element(radius, i) > outer_radius) {
          typename ExteriorVarsComputer::Cache cache{1};
          ExteriorVarsComputer computer{
              x_i, get_element(radius, i), radial_solution_,
              *equation_of_state_,
              std::forward<VarsComputerArgs>(vars_computer_args)...};
          expand_pack(get_var(i, cache, computer, Tags{})...);
        } else if (get_element(radius, i) > center_radius_cutoff) {
          typename InteriorVarsComputer::Cache cache{1};
          InteriorVarsComputer computer{
              x_i, get_element(radius, i), radial_solution_,
              *equation_of_state_,
              std::forward<VarsComputerArgs>(vars_computer_args)...};
          expand_pack(get_var(i, cache, computer, Tags{})...);
        } else {
          typename CenterVarsComputer::Cache cache{1};
          CenterVarsComputer computer{
              x_i, get_element(radius, i), radial_solution_,
              *equation_of_state_,
              std::forward<VarsComputerArgs>(vars_computer_args)...};
          expand_pack(get_var(i, cache, computer, Tags{})...);
        }
      }
      return vars;
    }
  }

 public:
  template <typename DataType>
  using tags = tmpl::list_difference<
      typename tov_detail::TovVariablesCache<DataType>::tags_list,
      tmpl::list<
          // Remove internal tags, which may not be available in the full domain
          tov_detail::Tags::MassOverRadius<DataType>,
          tov_detail::Tags::LogSpecificEnthalpy<DataType>,
          tov_detail::Tags::ConformalFactor<DataType>,
          tov_detail::Tags::DrConformalFactor<DataType>,
          tov_detail::Tags::ArealRadius<DataType>,
          tov_detail::Tags::DrArealRadius<DataType>,
          tov_detail::Tags::DrPressure<DataType>,
          tov_detail::Tags::MetricTimePotential<DataType>,
          tov_detail::Tags::DrMetricTimePotential<DataType>,
          tov_detail::Tags::MetricRadialPotential<DataType>,
          tov_detail::Tags::DrMetricRadialPotential<DataType>,
          tov_detail::Tags::MetricAngularPotential<DataType>,
          tov_detail::Tags::DrMetricAngularPotential<DataType>>>;

 private:
  friend bool operator==(const TovStar& lhs, const TovStar& rhs);

  double central_rest_mass_density_ =
      std::numeric_limits<double>::signaling_NaN();
  std::unique_ptr<equation_of_state_type> equation_of_state_;
  RelativisticEuler::Solutions::TovCoordinates coordinate_system_{};
  RelativisticEuler::Solutions::TovSolution radial_solution_{};
};

bool operator!=(const TovStar& lhs, const TovStar& rhs);
}  // namespace RelativisticEuler::Solutions
