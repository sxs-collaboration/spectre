// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::AnalyticData {
namespace detail {
/*!
 * \brief Read a massive star supernova progenitor from file.
 */
class ProgenitorProfile {
 public:
  ProgenitorProfile() = default;

  ProgenitorProfile(const std::string& filename);
  // interpolate function
  std::array<double, 6> interpolate(double target_radius,
                                    double target_cos_theta,
                                    bool interpolate_hydro_vars) const;

  double polytropic_index() const { return polytropic_index_; }

  double max_radius() const { return maximum_radius_; }

  // Change private density ratio based on user-defined value,
  // accessed in CCSNCollapse() call
  void set_dens_ratio(double max_dens_ratio) {
    max_density_ratio_for_linear_interpolation_ = max_dens_ratio;
  }

  void pup(PUP::er& p);

 private:
  double maximum_radius_{std::numeric_limits<double>::signaling_NaN()};
  double max_density_ratio_for_linear_interpolation_{
      std::numeric_limits<double>::signaling_NaN()};
  double polytropic_index_{std::numeric_limits<double>::signaling_NaN()};
  size_t num_radial_points_{std::numeric_limits<size_t>::max()};
  size_t num_angular_points_{std::numeric_limits<size_t>::max()};
  size_t num_grid_points_{std::numeric_limits<size_t>::max()};

  // Begin constants to translate units from cgs to c=G=M_Sun=(k_Boltzmann=1)=1
  // Divide cm/s by speed_of_light_cgs_ to get into c=G=M_Sun=1 units
  static constexpr double speed_of_light_cgs_ = 2.99792458e10;

  // Multiply cm by c2g_length_ to get into c=G=M_Sun=1 units
  // (speed of light)^2 / (Newton's constant * Msun)
  // c2g_length_ = c^2 / (G*M_Sun)
  // Note G*M_Sun = 1.32712440042x10^26 +/- 1e16 [1/cm]
  // G*M_Sun factor from https://doi.org/10.1063/1.4921980
  // c2g_length_~6.7706e-6 [1/cm]
  static constexpr double g_times_m_sun_ = 1.32712440042e26;
  static constexpr double c2g_length_ =
      square(speed_of_light_cgs_) / g_times_m_sun_;

  // Multiply g/cm^3 by c2g_dens_ to get into c=G=M_Sun=1 units
  // c2g_dens_ = (G*M_Sun)^2*G/c^6.
  // Note G = 6.67430x10^-8.
  // While G is not known to as many significant figures as G*M_Sun,
  // it's uncertainty is still less than that of the maximum mass of
  // a neutron star (~1%~10%), a quantity assumed to be 2.2 M_Sun when
  // converting to c=G=M_Sun=1 units; see
  // https://iopscience.iop.org/article/10.3847/2041-8213/aaa401.
  // G factor from
  // https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.93.025010.
  // c2g_dens_~1.61887e-18 [cm^3/g]
  static constexpr double newtons_grav_constant_ = 6.67430e-8;
  static constexpr double c2g_dens_ = square(g_times_m_sun_) *
                                      newtons_grav_constant_ /
                                      pow<6>(speed_of_light_cgs_);

  // Multiply Kelvin by c2g_temp_ to get into c=G=M_Sun=k_Boltzmann=1 units.
  // c2g_temp_ = (k_B/(M_Sun*c^2))
  // Note k_B = 1.380649e−16.
  // k_B factor from
  // https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.93.025010.
  // c2g_temp_=7.72567e-71
  static constexpr double boltzmann_constant_ = 1.380649e-16;
  // Note setting MSun=1.0 is a choice made independent of EoS,
  // so that there are no unit ambiguities between 2 simulations
  // ran with different EoSs (e.g., polytropic vs. tabulated)
  static constexpr double m_sun_ = g_times_m_sun_ / newtons_grav_constant_;
  static constexpr double c2g_temp_ =
      boltzmann_constant_ / (m_sun_ * square(speed_of_light_cgs_));

  // readin data
  DataVector radius_;
  DataVector rest_mass_density_;
  DataVector fluid_velocity_;
  DataVector electron_fraction_;
  DataVector chi_;
  DataVector metric_potential_;
  DataVector temperature_;
};
}  // namespace detail

/*!
 * \brief Evolve a stellar collapse (progenitor) profile through collapse.
 *
 * Read the 1D core-collapse supernova (CCSN) profile from file, assuming
 * the following format of the data:
 * * 1st row - number of radial points in data set, followed by 6 columns of 0s
 * * Subsequent rows contain these column headers:
 *   * radius, rest mass density, specific internal energy, radial velocity,
 *     electron fraction, \f$X\f$, \f$\Phi\f$.
 *
 * These columns assume a metric of the form \f$ds^2 = -e^{2\Phi} dt^2 +
 * X^2 dr^2 + r^2 d\theta^2 + r^2 \sin^2\theta d\phi^2\f$, following
 * \cite Oconnor2015.
 *
 * Units:
 * The input file is assumed to be in cgs units and
 * will be converted to dimensionless units upon
 * readin, setting \f$c = G = M_\odot = 1\f$.
 *
 * Rotation:
 * Once the stationary progenitor is constructed, an artificial
 * rotation profile can be assigned to the matter.  It takes the form
 * \f$\Omega(r) = \Omega_0/(1 + (r_\perp/A)^2)\f$, where \f$\Omega_0\f$ is
 * the central rotation rate, \f$r_\perp\f$ is the distance from the rotation
 * axis, and \f$A\f$ is the differential rotation parameter.  Large values
 * of \f$A\f$ correspond to more solid-body rotation.  Small values correspond
 * to stronger differential rotation.  The axis of rotation is assumed to
 * be the z axis.
 *
 */
class CcsnCollapse : public virtual evolution::initial_data::InitialData,
                     public MarkAsAnalyticData,
                     public AnalyticDataBase {
  template <typename DataType>
  struct IntermediateVariables {
    IntermediateVariables(
        const tnsr::I<DataType, 3, Frame::Inertial>& in_coords,
        double in_delta_r);

    struct MetricData {
      MetricData() = default;
      MetricData(size_t num_points);
      DataType chi;
      DataType metric_potential;
    };

    const tnsr::I<DataType, 3, Frame::Inertial>& coords;
    DataType radius;
    DataType phi;
    DataType cos_theta;
    DataType sin_theta;
    double delta_r;

    std::optional<DataType> rest_mass_density;
    std::optional<DataType> fluid_velocity;
    std::optional<DataType> electron_fraction;
    std::optional<DataType> temperature;
    std::optional<MetricData> metric_data;
    // Data for 2nd-order FD derivatives.
    // metric info at different indices for finite differences
    std::optional<std::array<MetricData, 3>> metric_data_upper{};
    std::optional<std::array<MetricData, 3>> metric_data_lower{};

    std::array<DataType, 3> radius_upper{};
    std::array<DataType, 3> radius_lower{};
    std::array<DataType, 3> cos_theta_upper{};
    std::array<DataType, 3> cos_theta_lower{};
    std::array<DataType, 3> sin_theta_upper{};
    std::array<DataType, 3> sin_theta_lower{};
    std::array<DataType, 3> phi_upper{};
    std::array<DataType, 3> phi_lower{};
  };

 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;

  /// The massive star progenitor data file.
  struct ProgenitorFilename {
    using type = std::string;
    static constexpr Options::String help = {
        "The supernova progenitor data file."};
  };

  /// The polytropic constant of the fluid.
  ///
  /// The remaining hydrodynamic primitive variables (e.g., pressure)
  /// will be calculated based on this \f$K\f$ for \f$P=K\rho^{\Gamma}\f$.
  struct PolytropicConstant {
    using type = double;
    static constexpr Options::String help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() { return 0.; }
  };

  /// Adiabatic index of the system at readin.
  ///
  /// Note the density profile used is from the initial profile calculated from
  /// the TOV equations, specified by ProgenitorFilename.  A lower, user
  /// defined, \f$\Gamma\f$ will cause a lower pressure for an equation of state
  /// of the form \f$P=K\rho^{\Gamma}\f$.  This effect triggers collapse for
  /// simplified CCSN models.
  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index that will trigger collapse."};
    static type lower_bound() { return 1.0; }
  };

  /// Central angular velocity artificially assigned at readin.
  ///
  /// Currently limited by the angular velocity of an object rotating in a
  /// circle, with radius = 1 cm (well below grid resolution) and
  /// transverse velocity = the speed of light.
  ///
  /// Due to effects from magnetic breaking, more realistic central angular
  /// velocities for massive stars are of order 0.1 [radians/sec] ~ 4.93e-7
  /// [code units] (see Figure 1 of \cite Pajkos2019 based on results from
  /// \cite Heger2005, or see \cite Richers2017 for an exploration of
  /// different values).
  struct CentralAngularVelocity {
    using type = double;
    static constexpr Options::String help = {
        "Central angular velocity of progenitor"};

    static type upper_bound() { return 147670.0; }
    static type lower_bound() { return -147670.0; }
  };

  /// Differential rotation parameter for artificially assigned
  /// rotation profile.
  struct DifferentialRotationParameter {
    using type = double;
    static constexpr Options::String help = {
        "Differential rotation parameter (large"
        " indicates solid body, small very differential)"};
    // This is ~1 cm, well below simulation resolution.
    // The lower bound is used to ensure a nonzero divisor.
    static type lower_bound() { return 6.7706e-6; }
  };

  /// Maximum density ratio for linear interpolation.
  ///
  /// If the ratio of two neighboring densities are greater than this value,
  /// then interpolation onto the SpECTRE grid reverts to linear, to avoid
  /// oscillations.  A ratio of 100 is used for the rotating TOV star test
  /// and a safe value to use if unsure.
  struct MaxDensityRatioForLinearInterpolation {
    using type = double;
    static constexpr Options::String help = {
        "If the ratio between neighboring density points is greater"
        " than this parameter, fall back to linear interpolation"
        " onto the SpECTRE grid."};
    static type lower_bound() { return 0.0; }
  };

  using options =
      tmpl::list<ProgenitorFilename, PolytropicConstant, AdiabaticIndex,
                 CentralAngularVelocity, DifferentialRotationParameter,
                 MaxDensityRatioForLinearInterpolation>;
  static constexpr Options::String help = {
      "Core collapse supernova initial data, read in from a profile containing"
      " hydrodynamic primitives and metric variables.  The data "
      "are read in from disk."};

  CcsnCollapse() = default;
  CcsnCollapse(const CcsnCollapse& /*rhs*/) = default;
  CcsnCollapse& operator=(const CcsnCollapse& /*rhs*/) = default;
  CcsnCollapse(CcsnCollapse&& /*rhs*/) = default;
  CcsnCollapse& operator=(CcsnCollapse&& /*rhs*/) = default;
  ~CcsnCollapse() override = default;

  CcsnCollapse(std::string progenitor_filename, double polytropic_constant,
               double adiabatic_index, double central_angular_velocity,
               double diff_rot_parameter, double max_dens_ratio_interp);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit CcsnCollapse(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CcsnCollapse);
  /// \endcond

  /// Retrieve a collection of variables at `(x)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    IntermediateVariables<DataType> intermediate_vars{
        x, 1.0e-4 * prog_data_.max_radius()};
    return {get<Tags>(variables(make_not_null(&intermediate_vars), x,
                                tmpl::list<Tags>{}))...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

 protected:
  template <typename DataType>
  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                                   Frame::Inertial>;

  template <typename DataType>
  using DerivShift = ::Tags::deriv<gr::Tags::Shift<DataType, 3>,
                                   tmpl::size_t<3>, Frame::Inertial>;

  template <typename DataType>
  using DerivSpatialMetric = ::Tags::deriv<gr::Tags::SpatialMetric<DataType, 3>,
                                           tmpl::size_t<3>, Frame::Inertial>;

  template <typename DataType>
  void interpolate_vars_if_necessary(
      gsl::not_null<IntermediateVariables<DataType>*> vars) const;

  template <typename DataType>
  void interpolate_deriv_vars_if_necessary(
      gsl::not_null<IntermediateVariables<DataType>*> vars) const;

  template <typename DataType>
  Scalar<DataType> lapse(const DataType& metric_potential) const;

  template <typename DataType>
  tnsr::I<DataType, 3, Frame::Inertial> shift(const DataType& radius) const;

  template <typename DataType>
  tnsr::ii<DataType, 3, Frame::Inertial> spatial_metric(
      const DataType& chi, const DataType& cos_theta, const DataType& sin_theta,
      const DataType& phi) const;

  template <typename DataType>
  tnsr::II<DataType, 3, Frame::Inertial> inverse_spatial_metric(
      const DataType& chi, const DataType& cos_theta, const DataType& sin_theta,
      const DataType& phi) const;

  template <typename DataType>
  Scalar<DataType> sqrt_det_spatial_metric(const DataType& chi) const;

  template <typename DataType>
  auto make_metric_data(size_t num_points) const ->
      typename IntermediateVariables<DataType>::MetricData;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(
      gsl::not_null<IntermediateVariables<DataType>*> vars,
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  // Will be added once temperature is used in grmhd tags
  // template <typename DataType>
  // auto variables(
  //     gsl::not_null<IntermediateVariables<DataType>*> vars,
  //     const tnsr::I<DataType, 3>& x,
  //     tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
  //     -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      gsl::not_null<IntermediateVariables<DataType>*> vars,
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::Shift<DataType, 3>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::Shift<DataType, 3>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::SpatialMetric<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<gr::Tags::SpatialMetric<DataType, 3>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>>;

  template <typename DataType>
  auto variables(
      gsl::not_null<IntermediateVariables<DataType>*> vars,
      const tnsr::I<DataType, 3>& x,
      tmpl::list<gr::Tags::InverseSpatialMetric<DataType, 3>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<DataType, 3>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<DerivLapse<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<DerivLapse<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<DerivShift<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<DerivShift<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<DerivSpatialMetric<DataType>>;

  template <typename DataType>
  auto variables(gsl::not_null<IntermediateVariables<DataType>*> vars,
                 const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::ExtrinsicCurvature<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<DataType, 3>>;

  friend bool operator==(const CcsnCollapse& lhs, const CcsnCollapse& rhs);

  std::string progenitor_filename_{};
  detail::ProgenitorProfile prog_data_{};
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  double central_angular_velocity_ =
      std::numeric_limits<double>::signaling_NaN();
  double inv_diff_rot_parameter_ = std::numeric_limits<double>::signaling_NaN();
};
bool operator!=(const CcsnCollapse& lhs, const CcsnCollapse& rhs);

}  // namespace grmhd::AnalyticData
