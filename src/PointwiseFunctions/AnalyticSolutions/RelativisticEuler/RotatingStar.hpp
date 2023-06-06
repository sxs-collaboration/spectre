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
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Solutions.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace RelativisticEuler::Solutions {
namespace detail {
/*!
 * \brief Read the CST rotating neutron star solution from file, rescaling the
 * solution assuming a polytropic constant `kappa`
 *
 * The CST RotNS code uses `kappa=1`.
 */
class CstSolution {
 public:
  CstSolution() = default;
  CstSolution(const std::string& filename, double kappa);

  std::array<double, 6> interpolate(double target_radius,
                                    double target_cos_theta,
                                    bool interpolate_hydro_vars) const;

  double polytropic_index() const { return polytropic_index_; }

  double equatorial_radius() const { return equatorial_radius_; }

  void pup(PUP::er& p);

 private:
  double maximum_radius_{std::numeric_limits<double>::signaling_NaN()};
  double max_density_ratio_for_linear_interpolation_ = 1.0e2;

  double equatorial_radius_{std::numeric_limits<double>::signaling_NaN()};
  double polytropic_index_{std::numeric_limits<double>::signaling_NaN()};
  double central_angular_speed_{std::numeric_limits<double>::signaling_NaN()};
  double rotation_profile_{std::numeric_limits<double>::signaling_NaN()};
  size_t num_radial_points_{std::numeric_limits<size_t>::max()};
  size_t num_angular_points_{std::numeric_limits<size_t>::max()};
  size_t num_grid_points_{std::numeric_limits<size_t>::max()};

  DataVector radius_;
  DataVector cos_theta_;  // Note that cos(theta) is between 0 and 1.
  DataVector rest_mass_density_;
  DataVector fluid_velocity_;
  DataVector alpha_;
  DataVector rho_;
  DataVector gamma_;
  DataVector omega_;
};
}  // namespace detail

/*!
 * \brief A solution obtained by reading in rotating neutron star initial data
 * from the RotNS code based on \cite Cook1992 and \cite Cook1994.
 *
 * The code that generates the initial data is part of a private SXS repository
 * called `RotNS`.
 *
 * The metric in spherical coordinates is given by \cite Cook1992
 *
 * \f{align}{
 *   ds^2=-e^{\gamma+\rho}dt^2+e^{2\alpha}(dr^2+r^2d\theta^2)
 *   +e^{\gamma-\rho}r^2\sin^2(\theta)(d\phi-\omega dt)^2.
 * \f}
 *
 * We use rotation about the \f$z\f$-axis. That is,
 *
 * \f{align}{
 *   g_{tt}
 *   &=-e^{\gamma+\rho} + e^{\gamma-\rho}r^2\sin^2(\theta)\omega^2 \\
 *   g_{rr}
 *   &=e^{2\alpha} \\
 *   g_{\theta\theta}
 *   &=e^{2\alpha}r^2 \\
 *   g_{\phi\phi}
 *   &=e^{\gamma-\rho}r^2\sin^2(\theta) \\
 *   g_{t\phi}
 *   &=-e^{\gamma-\rho}r^2\sin^2(\theta)\omega.
 * \f}
 *
 * We can transform from spherical to Cartesian coordinates using
 *
 * \f{align}{
 *   \label{eq:Jacobian}
 *   \frac{\partial (r,\theta,\phi)}{\partial (x,y,z)}=
 *   \begin{pmatrix}
 *     \cos(\phi) \sin(\theta) & \sin(\theta)\sin(\phi) & \cos(\theta) \\
 *     \tfrac{\cos(\phi)\cos(\theta)}{r} & \tfrac{\sin(\phi)\cos(\theta)}{r}
 *     & -\tfrac{\sin(\theta)}{r} \\
 *     -\tfrac{\sin(\phi)}{r\sin(\theta)} & \tfrac{\cos(\phi)}{r\sin(\theta)}
 *     & 0
 *   \end{pmatrix}
 * \f}
 *
 * and
 *
 * \f{align}{
 *   \frac{\partial (x,y,z)}{\partial (r,\theta,\phi)}=
 *   \begin{pmatrix}
 *     \cos(\phi) \sin(\theta) & r\cos(\phi)\cos(\theta) &
 * -r\sin(\theta)\sin(\phi)
 *     \\
 *     \sin(\theta)\sin(\phi) & r\sin(\phi)\cos(\theta) &
 * r\cos(\phi)\sin(\theta)
 *     \\
 *     \cos(\theta) & -r\sin(\theta) & 0
 *   \end{pmatrix}
 * \f}
 *
 *
 * We denote the lapse as \f$N\f$ since \f$\alpha\f$ is already being used,
 *
 * \f{align}{
 *   N = e^{(\gamma+\rho)/2}.
 * \f}
 *
 * The shift is
 *
 * \f{align}{
 *   \beta^\phi =& -\omega
 * \f}
 *
 * so
 *
 * \f{align}{
 *   \beta^x &= \partial_\phi x \beta^\phi = \sin(\phi)\omega r \sin(\theta), \\
 *   \beta^y &= \partial_\phi y \beta^\phi = -\cos(\phi)\omega r \sin(\theta),
 * \\ \beta^z &= 0. \f}
 *
 * The spatial metric is
 *
 * \f{align}{
 *   \gamma_{xx}
 *   &= \sin^2(\theta)\cos^2(\phi) e^{(2 \alpha)}
 *      + \cos^2(\theta)\cos^2(\phi) e^{(2 \alpha)}
 *      + \sin^2(\phi) e^{(\gamma-\rho)} \notag \\
 *   &=\cos^2(\phi)e^{2\alpha} + \sin^2(\phi) e^{(\gamma-\rho)} \\
 *   \gamma_{xy}
 *   &= \sin^2(\theta)\cos(\phi)\sin(\phi) e^{(2 \alpha)}
 *      + \cos^2(\theta)\cos(\phi)\sin(\phi) e^{(2 \alpha)}
 *      - \sin(\phi)\cos(\phi) e^{(\gamma-\rho)} \notag \\
 *   &=\cos(\phi)\sin(\phi)e^{2\alpha} - \sin(\phi)\cos(\phi) e^{(\gamma-\rho)}
 * \\ \gamma_{yy}
 *   &= \sin^2(\theta)\sin^2(\phi) e^{(2 \alpha)}
 *      + \cos^2(\theta)\sin^2(\phi) e^{(2 \alpha)}
 *      + \cos^2(\phi) e^{(\gamma-\rho)} \notag \\
 *   &=\sin^2(\phi)e^{2\alpha} + \cos^2(\phi) e^{(\gamma-\rho)} \\
 *   \gamma_{xz}
 *   &= \sin(\theta)\cos(\phi)\cos(\theta) e^{2\alpha}
 *      -\cos(\theta)\cos(\phi)\sin(\theta)e^{2\alpha} = 0 \\
 *   \gamma_{yz}
 *   &= \sin(\theta)\sin(\phi)\cos(\theta) e^{2\alpha}
 *      -\cos(\theta)\sin(\phi)\sin(\theta)e^{2\alpha} = 0 \\
 *   \gamma_{zz}
 *   &=\cos^2(\theta)e^{2\alpha} + \sin^2(\theta) e^{2\alpha}
 *      = e^{2\alpha}
 * \f}
 *
 * and its determinant is
 *
 * \f{align}{
 *   \gamma = e^{4\alpha + (\gamma-\rho)} = e^{4\alpha}e^{(\gamma-\rho)}.
 * \f}
 *
 * At \f$r=0\f$ we have \f$2\alpha=\gamma-\rho\f$ and so the
 * \f$\gamma_{xx}=\gamma_{yy}=\gamma_{zz} = e^{2\alpha}\f$ and all other
 * components are zero. The inverse spatial metric is given by
 *
 * \f{align}{
 *   \gamma^{xx}
 *   &= \frac{\gamma_{yy}}{e^{2\alpha}e^{(\gamma-\rho)}} =
 *      \left[\sin^2(\phi)e^{2\alpha} + \cos^2(\phi) e^{(\gamma-\rho)}\right]
 *      e^{-2\alpha} e^{-(\gamma-\rho)} \notag \\
 *   &=\sin^2(\phi)e^{-(\gamma-\rho)} + \cos^2(\phi) e^{-2\alpha} \\
 *   \gamma^{yy}
 *   &= \frac{\gamma_{xx}}{e^{2\alpha}e^{(\gamma-\rho)}} =
 *      \left[\cos^2(\phi)e^{2\alpha} + \sin^2(\phi) e^{(\gamma-\rho)}\right]
 *      e^{-2\alpha} e^{-(\gamma-\rho)} \notag \\
 *   &=\cos^2(\phi) e^{-(\gamma-\rho)} + \sin^2(\phi) e^{-2\alpha} \\
 *   \gamma^{xy}
 *   &=\frac{-\gamma_{xy}}{e^{2\alpha}e^{(\gamma-\rho)}} =
 *     -\left[\cos(\phi)\sin(\phi)e^{2\alpha} - \sin(\phi)\cos(\phi)
 *      e^{(\gamma-\rho)}\right] e^{-2\alpha} e^{-(\gamma-\rho)} \notag \\
 *   &=-\cos(\phi)\sin(\phi)e^{-(\gamma-\rho)} -
 *     \sin(\phi)\cos(\phi)e^{-2\alpha} \notag \\
 *   &=\cos(\phi)\sin(\phi) \left[e^{-2\alpha} -
 *     e^{-(\gamma-\rho)}\right] \\
 *   \gamma^{xz}
 *   &= 0 \\
 *   \gamma^{yz}
 *   &= 0 \\
 *   \gamma^{zz} &= e^{-2\alpha}.
 * \f}
 *
 * The 4-velocity in spherical coordinates is given by
 *
 * \f{align}{
 *   u^{\bar{a}}=\frac{e^{-(\rho+\gamma)/2}}{\sqrt{1-v^2}}
 *   \left[1,0,0,\Omega\right],
 * \f}
 *
 * where
 *
 * \f{align}{
 *   v=(\Omega-\omega)r\sin(\theta)e^{-\rho}.
 * \f}
 *
 * Transforming to Cartesian coordinates we have
 *
 * \f{align}{
 *   u^t
 *   &=\frac{e^{-(\rho+\gamma)/2}}{\sqrt{1-v^2}} \\
 *   u^x
 *   &=\partial_\phi x u^\phi = -r\sin(\theta)\sin(\phi) u^t\Omega \\
 *   u^y
 *   &=\partial_\phi y u^\phi = r\sin(\theta)\cos(\phi) u^t\Omega \\
 *   u^z &= 0.
 * \f}
 *
 * The Lorentz factor is given by
 *
 * \f{align}{
 *   W
 *   &=Nu^t=e^{(\gamma+\rho)/2}\frac{e^{-(\rho+\gamma)/2}}{\sqrt{1-v^2}} \notag
 * \\
 *   &=\frac{1}{\sqrt{1-v^2}}.
 * \f}
 *
 * Using
 *
 * \f{align}{
 *   v^i = \frac{1}{N}\left(\frac{u^i}{u^t} + \beta^i\right)
 * \f}
 *
 * we get
 *
 * \f{align}{
 *   v^x
 *   &= -e^{-(\gamma+\rho)/2}r\sin(\theta)\sin(\phi)(\Omega-\omega)
 *   =-e^{-(\gamma-\rho)/2}\sin(\phi) v\\
 *   v^y
 *   &= e^{-(\gamma+\rho)/2}r\sin(\theta)\cos(\phi)(\Omega-\omega)
 *   = e^{-(\gamma-\rho)/2}\cos(\phi)v \\
 *   v^z&=0.
 * \f}
 *
 * Lowering with the spatial metric we get
 *
 * \f{align}{
 *   v_x
 *   &=\gamma_{xx} v^x + \gamma_{xy} v^y \notag \\
 *   &=-\left[\cos^2(\phi)e^{2\alpha}+\sin^2(\phi)e^{\gamma-\rho}\right]
 *     e^{-(\gamma-\rho)/2}\sin(\phi) v \notag \\
 *   &+\left[\cos(\phi)\sin(\phi)e^{2\alpha} -
 *     \sin(\phi)\cos(\phi)e^{\gamma-\rho}\right]
 *     e^{-(\gamma-\rho)/2}\cos(\phi)v \notag \\
 *   &=-e^{-(\gamma-\rho)/2}v\sin(\phi)e^{\gamma-\rho}
 *     \left[\sin^2(\phi)+\cos^2(\phi)\right] \notag \\
 *   &=-e^{(\gamma-\rho)/2}v\sin(\phi) \\
 *   v_y
 *   &=\gamma_{yx} v^x + \gamma_{yy} v^y \notag \\
 *   &=-\left[\cos(\phi)\sin(\phi)e^{2\alpha} - \sin(\phi)\cos(\phi)
 *     e^{(\gamma-\rho)}\right] e^{-(\gamma-\rho)/2}\sin(\phi) v \notag \\
 *   &+\left[\sin^2(\phi)e^{2\alpha} + \cos^2(\phi) e^{(\gamma-\rho)}\right]
 *     e^{-(\gamma-\rho)/2}\cos(\phi)v \notag \\
 *   &=e^{(\gamma-\rho)/2}v\cos(\phi) \\
 *   v_z &= 0.
 * \f}
 *
 * This is consistent with the Lorentz factor read off from \f$u^t\f$ since
 * \f$v^iv_i=v^2\f$. For completeness, \f$u_i=Wv_i\f$ so
 *
 * \f{align}{
 *   u_x
 *   &=-\frac{e^{(\gamma-\rho)/2}v\sin(\phi)}{\sqrt{1-v^2}} \\
 *   u_y
 *   &=\frac{e^{(\gamma-\rho)/2}v\cos(\phi)}{\sqrt{1-v^2}} \\
 *   u_z&=0.
 * \f}
 *
 * \warning Near (within `1e-2`) \f$r=0\f$ the numerical errors from
 * interpolation and computing the metric derivatives by finite difference no
 * longer cancel out and so the `tilde_s` time derivative only vanishes to
 * roughly `1e-8` rather than machine precision. Computing the Cartesian
 * derivatives from analytic differentiation of the radial and angular
 * polynomial fits might improve the situation but is a decent about of work to
 * implement.
 */
class RotatingStar : public virtual evolution::initial_data::InitialData,
                     public MarkAsAnalyticSolution,
                     public AnalyticSolution<3> {
  template <typename DataType>
  struct IntermediateVariables {
    IntermediateVariables(
        const tnsr::I<DataType, 3, Frame::Inertial>& in_coords,
        double in_delta_r);

    struct MetricData {
      MetricData() = default;
      MetricData(size_t num_points);
      DataType alpha;
      DataType rho;
      DataType gamma;
      DataType omega;
    };

    const tnsr::I<DataType, 3, Frame::Inertial>& coords;
    DataType radius;
    DataType phi;
    DataType cos_theta;
    DataType sin_theta;
    // Data at coords
    std::optional<DataType> rest_mass_density;
    std::optional<DataType> fluid_velocity;
    std::optional<MetricData> metric_data;
    // Data for 2nd-order FD derivatives.
    //
    // Technically we could do full-order accurate derivatives by using
    // Jacobians to transform quantities, but since we really want the initial
    // data to be solved natively in SpECTRE and satisfy the Einstein equations
    // with neutrinos and magnetic fields, doing the 2nd order FD like SpEC
    // should be fine.
    //
    // Note: We guard all the upper/lower values by checking if
    // metric_data_upper is computed. While not perfect, this simplifies the
    // code a lot.
    double delta_r;
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

  /// The path to the RotNS data file.
  struct RotNsFilename {
    using type = std::string;
    static constexpr Options::String help = {
        "The path to the RotNS data file."};
  };

  /// The polytropic constant of the fluid.
  ///
  /// The data in the RotNS file will be rescaled.
  struct PolytropicConstant {
    using type = double;
    static constexpr Options::String help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() { return 0.; }
  };

  using options = tmpl::list<RotNsFilename, PolytropicConstant>;
  static constexpr Options::String help = {
      "Rotating neutron star initial data solved by the RotNS solver. The data "
      "is read in from disk."};

  RotatingStar() = default;
  RotatingStar(const RotatingStar& /*rhs*/) = default;
  RotatingStar& operator=(const RotatingStar& /*rhs*/) = default;
  RotatingStar(RotatingStar&& /*rhs*/) = default;
  RotatingStar& operator=(RotatingStar&& /*rhs*/) = default;
  ~RotatingStar() override = default;

  RotatingStar(std::string rot_ns_filename, double polytropic_constant);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit RotatingStar(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RotatingStar);
  /// \endcond

  /// Retrieve a collection of variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         const double /*t*/,
                                         tmpl::list<Tags...> /*meta*/) const {
    IntermediateVariables<DataType> intermediate_vars{
        x, 1.0e-4 * cst_solution_.equatorial_radius()};
    return {get<Tags>(variables(make_not_null(&intermediate_vars), x,
                                tmpl::list<Tags>{}))...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

  double equatorial_radius() const {
    return cst_solution_.equatorial_radius();
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
  Scalar<DataType> lapse(const DataType& gamma, const DataType& rho) const;

  template <typename DataType>
  tnsr::I<DataType, 3, Frame::Inertial> shift(const DataType& omega,
                                              const DataType& phi,
                                              const DataType& radius,
                                              const DataType& sin_theta) const;

  template <typename DataType>
  tnsr::ii<DataType, 3, Frame::Inertial> spatial_metric(
      const DataType& gamma, const DataType& rho, const DataType& alpha,
      const DataType& phi) const;

  template <typename DataType>
  tnsr::II<DataType, 3, Frame::Inertial> inverse_spatial_metric(
      const DataType& gamma, const DataType& rho, const DataType& alpha,
      const DataType& phi) const;

  template <typename DataType>
  Scalar<DataType> sqrt_det_spatial_metric(const DataType& gamma,
                                           const DataType& rho,
                                           const DataType& alpha) const;

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

  friend bool operator==(const RotatingStar& lhs, const RotatingStar& rhs);

  std::string rot_ns_filename_{};
  detail::CstSolution cst_solution_{};
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
};

bool operator!=(const RotatingStar& lhs, const RotatingStar& rhs);
}  // namespace RelativisticEuler::Solutions
