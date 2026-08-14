// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <brigand/brigand.hpp>

#include <array>
#include <functional>
#include <optional>
#include <utility>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::AnalyticData {

namespace detail {

template <typename DataType>
using BinaryWithGravitationalWavesVariablesCache =
    cached_temp_buffer_from_typelist<tmpl::append<
        common_tags<DataType>,
        tmpl::list<
            ::Tags::deriv<
                Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
                tmpl::size_t<3>, Frame::Inertial>,
            gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
            gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
            gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0>,
            // For initial guesses
            Xcts::Tags::ConformalFactorMinusOne<DataType>,
            Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType>,
            Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>,
        hydro_tags<DataType>>>;

template <typename DataType>
struct BinaryWithGravitationalWavesVariables
    : CommonVariables<DataType,
                      BinaryWithGravitationalWavesVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = BinaryWithGravitationalWavesVariablesCache<DataType>;
  using Base =
      CommonVariables<DataType,
                      BinaryWithGravitationalWavesVariablesCache<DataType>>;
  using Base::operator();

  using superposed_tags = tmpl::append<
      tmpl::list<
          Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
          gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
          gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
          gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0>>,
      hydro_tags<DataType>>;

  using boost_tags = tmpl::append<
      tmpl::list<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                 Xcts::Tags::ConformalFactorMinusOne<DataType>,
                 Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType>,
                 Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial>>>;

  BinaryWithGravitationalWavesVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataVector, Dim>& local_x,
      const std::array<double, 2>& local_masses,
      const std::array<double, 2>& local_xcoords,
      const std::array<double, 3>& local_momentum_left,
      const std::array<double, 3>& local_momentum_right,
      const double local_y_offset, const double local_z_offset,
      const double local_attenuation_parameter,
      const double local_attenuation_radius,
      std::array<tnsr::I<DataVector, Dim>, 2> local_x_isolated,
      tuples::tagged_tuple_from_typelist<superposed_tags> local_flat_vars,
      std::array<tuples::tagged_tuple_from_typelist<superposed_tags>, 2>
          local_isolated_vars,
      std::array<tuples::tagged_tuple_from_typelist<boost_tags>, 2>
          local_boost_vars)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        mesh(std::move(local_mesh)),
        inv_jacobian(std::move(local_inv_jacobian)),
        x(local_x),
        mass_left(local_masses[0]),
        mass_right(local_masses[1]),
        xcoords(local_xcoords),
        momentum_left(local_momentum_left),
        momentum_right(local_momentum_right),
        offset({local_y_offset, local_z_offset}),
        attenuation_parameter(local_attenuation_parameter),
        attenuation_radius(local_attenuation_radius),
        x_isolated(std::move(local_x_isolated)),
        flat_vars(std::move(local_flat_vars)),
        isolated_vars(std::move(local_isolated_vars)),
        boost_vars(std::move(local_boost_vars)) {}

  std::optional<std::reference_wrapper<const Mesh<Dim>>> mesh;
  std::optional<std::reference_wrapper<const InverseJacobian<
      DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
      inv_jacobian;

  tnsr::I<DataVector, Dim> x;
  double mass_left;
  double mass_right;
  std::array<double, 2> xcoords;
  std::array<double, 3> momentum_left;
  std::array<double, 3> momentum_right;
  std::array<double, 2> offset;
  double attenuation_parameter;
  double attenuation_radius;
  std::array<tnsr::I<DataVector, Dim>, 2> x_isolated;
  tuples::tagged_tuple_from_typelist<superposed_tags> flat_vars;
  std::array<tuples::tagged_tuple_from_typelist<superposed_tags>, 2>
      isolated_vars;
  std::array<tuples::tagged_tuple_from_typelist<boost_tags>, 2> boost_vars;

  double time_displacement = 0.01;
  DataType present_time = make_with_value<DataVector>(x, 1.);

  template <typename Tag,
            Requires<tmpl::list_contains_v<superposed_tags, Tag>> = nullptr>
  void superposition(gsl::not_null<typename Tag::type*> superposed_var,
                     gsl::not_null<Cache*> /*cache*/, Tag /*meta*/) const {
    for (size_t i = 0; i < superposed_var->size(); ++i) {
      (*superposed_var)[i] = get<Tag>(isolated_vars[0])[i] +
                             get<Tag>(isolated_vars[1])[i] -
                             get<Tag>(flat_vars)[i];
    }
  }

  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>
                      meta) const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>
          meta) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> meta) const override;
  void operator()(gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
                  gsl::not_null<Cache*> cache,
                  ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> meta)
      const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> /*cache*/,
      Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const override {
    std::fill(shift_background->begin(), shift_background->end(), 0.);
  }
  void operator()(
      gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
      gsl::not_null<Cache*> /*cache*/,
      ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
    std::fill(deriv_shift_background->begin(), deriv_shift_background->end(),
              0.);
  }
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const override;
  void operator()(
      const gsl::not_null<Scalar<DataType>*> conformal_energy_density,
      const gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> meta) const {
    superposition(conformal_energy_density, cache, meta);
  }
  void operator()(
      const gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
      const gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> meta) const {
    superposition(conformal_stress_trace, cache, meta);
  }
  void operator()(
      const gsl::not_null<tnsr::I<DataType, Dim>*> conformal_momentum_density,
      const gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> meta)
      const {
    superposition(conformal_momentum_density, cache, meta);
  }
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
                  gsl::not_null<Cache*> cache,
                  Xcts::Tags::ConformalFactorMinusOne<DataType> meta) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> meta) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> meta) const;
  void operator()(const gsl::not_null<Scalar<DataType>*> rest_mass_density,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::RestMassDensity<DataType> meta) const {
    superposition(rest_mass_density, cache, meta);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::SpecificEnthalpy<DataType> meta) const {
    superposition(specific_enthalpy, cache, meta);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> pressure,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::Pressure<DataType> meta) const {
    superposition(pressure, cache, meta);
  }
  void operator()(const gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::SpatialVelocity<DataType, 3> meta) const {
    superposition(spatial_velocity, cache, meta);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> lorentz_factor,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::LorentzFactor<DataType> meta) const {
    superposition(lorentz_factor, cache, meta);
  }
  void operator()(const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::MagneticField<DataType, 3> meta) const {
    superposition(magnetic_field, cache, meta);
  }

 private:
  // void interpolate_past_history();
  // DataType find_retarded_time_left(DataType t0) const;
  // DataType find_retarded_time_right(DataType t0) const;
  Scalar<DataType> get_t_distance_left(DataType t) const;
  Scalar<DataType> get_t_distance_right(DataType t) const;
  Scalar<DataType> get_t_separation(DataType t) const;
  tnsr::I<DataType, 3> get_t_momentum_left(DataType t) const;
  tnsr::I<DataType, 3> get_t_momentum_right(DataType t) const;
  tnsr::I<DataType, 3> get_t_normal_left(DataType t) const;
  tnsr::I<DataType, 3> get_t_normal_right(DataType t) const;
  tnsr::I<DataType, 3> get_t_normal_lr(DataType t) const;
  // DataType integrate_term(DataType t, size_t i, size_t j, int left_right,
  // double t0) const;
  Scalar<DataType> get_t_trace_extrinsic_curvature(
      DataType t, Mesh<3> local_mesh,
      InverseJacobian<DataType, 3, Frame::ElementLogical, Frame::Inertial>
          local_inv_jacobian) const;
  DataType get_t_attenuation_function(DataType t) const;
  tnsr::ii<DataType, 3> get_t_conformal_metric(DataType t) const;
  tnsr::ii<DataType, 3> get_t_radiative_term(DataType t) const;
  tnsr::ii<DataType, 3> get_t_near_zone_term(DataType t) const;
  tnsr::ii<DataType, 3> get_t_present_term(DataType t) const;
  tnsr::ii<DataType, 3> get_t_past_term(DataType t) const;
  tnsr::ii<DataType, 3> get_t_integral_term(DataType t) const;
  Scalar<DataType> get_t_lapse(DataType t) const;
  tnsr::I<DataType, 3> get_t_shift(DataType t) const;
  tnsr::aa<DataType, 3> get_t_boosted_spacetime_metric_left(DataType t) const;
  tnsr::aa<DataType, 3> get_t_boosted_spacetime_metric_right(DataType t) const;
  tnsr::aa<DataType, 3> get_t_superposed_spacetime_metric(DataType t) const;
};
}  // namespace detail

/*!
 * \brief   Binary initial data with realistic wave background,
 * constructed in Post-Newtonian approximations.
 *
 * The main goal of this implementation is to improve the extracted
 * wave forms, for example, by minimizing junk radiation.
 * The data is only valid for black holes without spin. Even so, there is some
 * work done to describe such systems that could later be implemented
 * \cite Steinhoff2008zr. The objects are constructed from a superposition of
 * two isolated objects that are boosted with respect to each other. The
 * radiative data is constructed from Post-Newtonian expansions for the
 * inspiral phase, in orders of \f$\epsilon = 1/c\f$, in \cite Jaranowski1997ky.
 * In ADMTT gauge it is possible to get the 3-metric as
 * \f$\gamma^{PN}_{ij} = \psi^{4}_{PN} \delta_{ij} + h^{TT}_{ij}\f$ where
 * \f$h^{TT}_{ij}\f$ is the radiative part and the non-radiative Post-Newtonian
 * conformal factor is given by
 *
 * \f{equation}{
 * \psi_{PN} = 1 + \sum_{a=1}^{2} \frac{E_a}{2 r_a} + O(\epsilon^6)
 * \f}
 *
 * and
 *
 * \f{equation}{E_a = (\epsilon^2) m_a + (\epsilon^4) \Bigr(\frac{p_a^2}{2 m_a}
 * - \frac{m_1 m_2}{2 r_{12}}\Bigr) \f}
 *
 * with \f$\vec{p}_a\f$ the linear momentum, \f$r_a\f$ the distance to each
 * black hole center of mass from the point of calculation and \f$r_{12}\f$
 * separation between the two black holes and \f$m_a\f$ is the mass of each
 * black hole. Near each black hole, the 3-metric can be approximated by the
 * Schwarzschild 3-metric in isotropic coordinates.
 *
 * In \cite Mundim2010hu, the radiative term \f$h^{TT}_{ij}\f$ is decomposed
 * into two parts, a near-zone that is only valid close to the black holes and a
 * remainder that makes corrections far from the black holes, \f$h^{TT}_{ij} =
 * h^{TT\ (NZ)}_{ij} + h^{TT\ (remainder)}_{ij} + O(\epsilon^5)\f$. The
 * near-zone term is given by $h^{TT\ (NZ)}_{ij} = (\epsilon^4) h^{TT}_{(4)ij} +
 * (\epsilon^5) h^{TT}_{(5)ij}$, with
 *
 * \f{align}{
 * h^{TT\ i j}_{(4)} &= \frac{1}{4} \sum_a \frac{1}{m_a r_a} \Bigr\{
 * [p_a^2-5(\hat{n}_a \cdot \vec{p}_a)^2] \delta^{i j}+2 p_a^i p_a^j
 * +[3(\hat{n}_a \cdot \vec{p}_a)^2-5p_a^2] n_a^i n_a^j +12(\hat{n}_a \cdot
 * \vec{p}_a) n_a^{(i} p_a^{j)} \Bigr\} \nonumber \\
 *  &+\frac{1}{8} \sum_a \sum_{b \neq a} m_a m_b \Bigr\{-\frac{32}{s_{a
 * b}}(\frac{1}{r_{a b}}+\frac{1}{s_{a b}}) n_{a b}^i n_{a
 * b}^j+2(\frac{r_a+r_b}{r_{a b}^3}+\frac{12}{s_{a b}^2}) n_a^i
 * n_b^j+32(\frac{2}{s_{a b}^2}-\frac{1}{r_{a b}^2}) n_a^{(i} n_{a b}^{j)}
 * \label{eq:near_zone_term} \\
 *  &+[\frac{5}{r_{a b} r_a}-\frac{1}{r_{a b}^3}(\frac{r_b^2}{r_a}+3
 * r_a)-\frac{8}{s_{a b}}(\frac{1}{r_a}+\frac{1}{s_{a b}})] n_a^i n_a^j+[5
 * \frac{r_a}{r_{a b}^3}(\frac{r_a}{r_b}-1)-\frac{17}{r_{a b} r_a}+\frac{4}{r_a
 * r_b}+\frac{8}{s_{a b}}(\frac{1}{r_a}+\frac{4}{r_{a b}})] \delta^{i j}\Bigr\},
 * \nonumber \f}
 *
 * where \f$\hat{n}_a\f$ is the unit normal vector pointing to the black hole
 * center of mass, \f$\hat{n}_{ab}\f$ is the unit normal vector pointing from
 * black hole \f$a\f$ to black hole \f$b\f$ and \f$s_{ab} = r_a + r_b +
 * r_{ab}\f$. The term \f$h^{TT}_{(5)ij}\f$ is a spatially constant field that
 * just varies in time, for initial data we can choose an initial time such that
 * \f$h^{TT}_{(5)ij} = 0\f$.
 *
 * Looking at \cite Kelly2007uc, the remainder term in itself is decomposed in
 * general computations for specific vectors as
 *
 * \f{equation}{
 * h^{TT\ (remainder)}_{ij} = H^{TT\ 1}_{ij} \Bigr[
 * \frac{\vec{p_1}}{\sqrt{m_1}}\Bigr] + H^{TT\ 2}_{ij} \Bigr[
 * \frac{\vec{p_2}}{\sqrt{m_2}}\Bigr] - H^{TT\ 1}_{ij} \Bigr[ \sqrt{\frac{m_1
 * m_2}{2 r_{12}}}  \hat{n_{12}} \Bigr] - H^{TT\ 2}_{ij} \Bigr[ \sqrt{\frac{m_1
 * m_2}{2 r_{12}}}  \hat{n_{12}} \Bigr], \f}
 *
 * each of this is composed of three different computations: one computed at
 * present time \f$t\f$, other at retarded time \f$t_{a}^{r}\f$ defined by \f$t
 * - t_{a}^{r} - r_a(t_{a}^{r}) = 0\f$ and the last is an integral between the
 * two times:
 *
 * \f{equation}{
 * H^{TT\ a}_{ij} [ \vec{u} ] = H^{TT\ a}_{ij} [ \vec{u} ; t] + H^{TT\ a}_{ij} [
 * \vec{u} ; t^{r}_a] + H^{TT\ a}_{ij} [ \vec{u} ; t_{a}^{r} \to t]. \f}
 *
 * Explicitly they are
 *
 * \f{equation}{
 * H^{TT\ a}_{ij} [ \vec{u} ; t] = -\frac{1}{4 r_a(t)} \Bigr\{ [u^2 - 5(\vec{u}
 * \cdot \hat{n}_a)^2] \delta_{ij} + 2 u^iu^j + 3(\vec{u}\cdot\hat{n}_a)^2 - 5
 * u^2] n_a^i n_a^j + 12 (\vec{u} \cdot \hat{n}_a) u^{(i}n_a^{j)}\Bigr\}_t,
 * \label{eq:present_term} \f}
 *
 * \f{equation}{
 * H^{TT\ a}_{ij} [ \vec{u} ; t^{r}_a] = \frac{1}{r_a(t^{r}_a)} \Bigr\{ [-2u^2
 * + 2(\vec{u} \cdot \hat{n}_a)^2] \delta^{ij} + 4u^iu^j + [2 u^2 + 2 (\vec{u}
 * \cdot \hat{n}_a)^2 ] n_a^i n_a^j - 8(\vec{u}\cdot\hat{n}_a) u^{(i}n_a^{j)}
 * \Bigr\}_{t^{r}_a},
 * \label{eq:retarded_term} \f}
 *
 * and
 *
 * \f{align}{
 * H^{TT\ a}_{ij} [ \vec{u} ; t^{r}_a \to t] &= \nonumber \\
 *  &- \int^t_{t^{r}_a} d\tau \frac{(t-\tau)}{r_a(\tau)^3}  \Bigr\{ [-5u^2 +
 * 9(\vec{u} \cdot \hat{n}_a)^2] \delta^{ij} + 6u^iu^j - 12
 * (\vec{u}\cdot\hat{n}_a) u^{(i}n_a^{j)} + [9 u^2 - 15(\vec{u} \cdot
 * \hat{n}_a)^2 ]  n_a^i n_a^j\Bigr\} \label{eq:integral_term} \\
 *  &- \int^t_{t^{r}_a} d\tau \frac{(t-\tau)^3}{r_a(\tau)^5}  \Bigr\{ [u^2 -
 * 5(\vec{u} \cdot \hat{n}_a)^2] \delta^{ij} + 2 u^iu^j - 20
 * (\vec{u}\cdot\hat{n}_a) u^{(i}n_a^{j)} + [-5 u^2 + 35(\vec{u} \cdot
 * \hat{n}_a)^2 ]  n_a^i n_a^j\Bigr\}. \nonumber \f}
 *
 * \warning The radiative terms, equations \f$\eqref{eq:near_zone_term}\f$,
 * \f$\eqref{eq:present_term}\f$, \f$\eqref{eq:retarded_term}\f$ and
 * \f$\eqref{eq:integral_term}\f$, are not implemented yet. Instead these terms
 * are set to zero.
 *
 * With this the whole spatial metric is computed up to \f$2PN\f$ order and the
 * radiative term agrees well with quadrupole predictions. In \cite Tichy2002ec,
 * the extrinsic curvature is given up to \f$2.5PN\f$ order by
 *
 * \f{equation}{
 * K^{ij}_{PN} = - \psi^{-10}_{PN} \Bigr[ (\epsilon^3) \tilde{\pi}_{(3)}^{ij} +
 * (\epsilon^5) \frac{1}{2} \dot{h}^{TT}_{(4)ij} + (\epsilon^5) (\phi_{(2)}
 * \tilde{\pi}_{(3)}^{ij})^{TT} \Bigr] + O(\epsilon^6). \f}
 *
 * where
 *
 * \f{equation}{
 *  \tilde{\pi}_{(3)}^{i j}=\frac{1}{16 \pi} \sum_a p_{a}^k\{-\delta_{i
 * j}(\frac{1}{r_a})_{, k}+2[\delta_{i k}(\frac{1}{r_a})_{, j}+\delta_{j
 * k}(\frac{1}{r_a})_{, i}]-\frac{1}{2} r_{a, i j k}\}. \f}
 *
 *
 * To be able to calculate equations \f$\eqref{eq:retarded_term}\f$ and
 * \f$\eqref{eq:integral_term}\f$ we need to look into the past history
 * of the binary at least up to the time were the generated wave can reach the
 * furthest point on the grid. To do so we must evolve the binary backward in
 * time. Because we are only looking into the inspiral phase we can follow a
 * simple Hamiltonian evolution computed in Post-Newtonian orders. The
 * equations to be solved are
 *
 * \f{equation}{
 * \frac{d X^i}{d t}=\frac{\partial H}{\partial P_i}
 * \f}
 *
 * and
 *
 * \f{equation}{
 * \frac{d P_i}{d t}=-\frac{\partial H}{\partial X^i}+F_i,
 * \f}
 *
 * where $H$ is the Post-Newtonian Hamiltonian, $X^i$ is the separation
 * vector between the two particles, $P_i$ is the momentum of one particle
 * in the center of mass frame and $F_i$ is the radiation-reaction flux term.
 * The Post-Newtonian Hamiltonian is given in \cite Buonanno2005xu.
 */
template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
class BinaryWithGravitationalWaves
    : public elliptic::analytic_data::Background,
      public elliptic::analytic_data::InitialGuess {
 public:
  struct XCoords {
    static constexpr Options::String help =
        "The coordinates on the x-axis where the two objects are placed.";
    using type = std::array<double, 2>;
  };
  struct Masses {
    static constexpr Options::String help =
        "The mass of each object, first left and second right.";
    using type = std::array<double, 2>;
  };
  struct MomentumLeft {
    static constexpr Options::String help =
        "The momentum assigned to the left object.";
    using type = std::array<double, 3>;
  };
  struct MomentumRight {
    static constexpr Options::String help =
        "The momentum assigned to the right object.";
    using type = std::array<double, 3>;
  };
  struct CenterOfMassOffset {
    static constexpr Options::String help = {
        "Offset in the y and z axes applied to both objects in order to "
        "control the center of mass."};
    using type = std::array<double, 2>;
  };
  struct ObjectLeft {
    static constexpr Options::String help =
        "The object placed on the negative x-axis.";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };
  struct ObjectRight {
    static constexpr Options::String help =
        "The object placed on the positive x-axis.";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };
  struct AttenuationParameter {
    static constexpr Options::String help =
        "The parameter controlling the transition width of the attenuation "
        "function.";
    using type = double;
  };
  struct AttenuationRadius {
    static constexpr Options::String help =
        "The parameter controlling the transition center of the attenuation "
        "function.";
    using type = double;
  };

  using options = tmpl::list<XCoords, Masses, MomentumLeft, MomentumRight,
                             CenterOfMassOffset, ObjectLeft, ObjectRight,
                             AttenuationParameter, AttenuationRadius>;
  static constexpr Options::String help =
      "Binary black hole initial data with realistic wave background, "
      "constructed in Post-Newtonian approximations. ";

  BinaryWithGravitationalWaves() = default;
  BinaryWithGravitationalWaves(const BinaryWithGravitationalWaves&) = delete;
  BinaryWithGravitationalWaves& operator=(const BinaryWithGravitationalWaves&) =
      delete;
  BinaryWithGravitationalWaves(BinaryWithGravitationalWaves&&) = default;
  BinaryWithGravitationalWaves& operator=(BinaryWithGravitationalWaves&&) =
      default;
  ~BinaryWithGravitationalWaves() override = default;

  BinaryWithGravitationalWaves(
      const std::array<double, 2> xcoords, const std::array<double, 2> masses,
      const std::array<double, 3> momentum_left,
      const std::array<double, 3> momentum_right,
      const std::array<double, 2> center_of_mass_offset,
      std::unique_ptr<IsolatedObjectBase> object_left,
      std::unique_ptr<IsolatedObjectBase> object_right,
      const double attenuation_parameter, const double attenuation_radius,
      const Options::Context& context = {})
      : xcoords_(xcoords),
        masses_(masses),
        momentum_left_(momentum_left),
        momentum_right_(momentum_right),
        y_offset_(center_of_mass_offset[0]),
        z_offset_(center_of_mass_offset[1]),
        superposed_objects_({std::move(object_left), std::move(object_right)}),
        attenuation_parameter_(attenuation_parameter),
        attenuation_radius_(attenuation_radius) {
    if (masses_[0] <= 0. || masses_[1] <= 0.) {
      PARSE_ERROR(context, "The masses must be positive.");
    }
    if (xcoords_[0] >= xcoords_[1]) {
      PARSE_ERROR(context, "Specify 'XCoords' ascending from left to right.");
    }
    if (attenuation_parameter_ < 0.) {
      PARSE_ERROR(context, "'AttenuationParameter' must be positive.");
    }
    if (attenuation_radius_ <= 0.) {
      PARSE_ERROR(context, "'AttenuationRadius' must be positive.");
    }
  }

  explicit BinaryWithGravitationalWaves(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m),
        elliptic::analytic_data::InitialGuess(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BinaryWithGravitationalWaves);

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataType>(x, std::nullopt, std::nullopt,
                                    tmpl::list<RequestedTags...>{});
  }
  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataVector>(x, mesh, inv_jacobian,
                                      tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    elliptic::analytic_data::Background::pup(p);
    elliptic::analytic_data::InitialGuess::pup(p);
    p | xcoords_;
    p | masses_;
    p | y_offset_;
    p | z_offset_;
    p | momentum_left_;
    p | momentum_right_;
    p | superposed_objects_;
    p | attenuation_parameter_;
    p | attenuation_radius_;
  }

  /// Coordinates of the objects, ascending left to right
  const std::array<double, 2>& x_coords() const { return xcoords_; }
  /// The momentum of the left object.
  const std::array<double, 3>& momentum_left() const { return momentum_left_; }
  /// The momentum of the right object.
  const std::array<double, 3>& momentum_right() const {
    return momentum_right_;
  }
  /// Offset in y and z coordinates of the objects
  double y_offset() const { return y_offset_; }
  double z_offset() const { return z_offset_; }
  /// The two objects. First entry is the left object, second entry is the right
  /// object.
  const std::array<std::unique_ptr<IsolatedObjectBase>, 2>& superposed_objects()
      const {
    return superposed_objects_;
  }
  double attenuation_parameter() const { return attenuation_parameter_; }
  double attenuation_radius() const { return attenuation_radius_; }

 private:
  std::array<double, 2> xcoords_{};
  std::array<double, 2> masses_{};
  std::array<double, 3> momentum_left_{};
  std::array<double, 3> momentum_right_{};
  double y_offset_{};
  double z_offset_{};
  std::array<std::unique_ptr<IsolatedObjectBase>, 2> superposed_objects_{};
  Xcts::Solutions::Flatness flatness_{};
  double attenuation_parameter_{};
  double attenuation_radius_{};

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<3>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    std::array<tnsr::I<DataVector, 3>, 2> x_isolated{{x, x}};
    const std::array<std::array<double, 3>, 2> coords_isolated{
        {{{xcoords_[0], y_offset_, z_offset_}},
         {{xcoords_[1], y_offset_, z_offset_}}}};
    // Possible optimization: Only retrieve those superposed tags from the
    // isolated solutions that are actually needed. This needs some dependency
    // logic, because some of the non-superposed tags depend on superposed tags.
    using VarsComputer =
        detail::BinaryWithGravitationalWavesVariables<DataType>;
    using requested_superposed_tags = typename VarsComputer::superposed_tags;
    std::array<tuples::tagged_tuple_from_typelist<requested_superposed_tags>, 2>
        isolated_vars;
    for (size_t i = 0; i < 2; ++i) {
      for (size_t dim = 0; dim < 3; dim++) {
        gsl::at(x_isolated, i).get(dim) -= gsl::at(coords_isolated, i)[dim];
      }
      gsl::at(isolated_vars, i) = get_isolated_vars<requested_superposed_tags>(
          *gsl::at(superposed_objects_, i), gsl::at(x_isolated, i));
    }
    auto flat_vars = flatness_.variables(x, requested_superposed_tags{});
    using requested_boost_tags = typename VarsComputer::boost_tags;
    std::array<tuples::tagged_tuple_from_typelist<requested_boost_tags>, 2>
        boost_vars;
    std::array<tnsr::I<DataVector, 3>, 2> x_unboosted{{x, x}};
    sr::lorentz_boost(make_not_null(&(gsl::at(x_unboosted, 0))),
                      gsl::at(x_isolated, 0), 0., momentum_left_ / masses_[0]);
    sr::lorentz_boost(make_not_null(&(gsl::at(x_unboosted, 1))),
                      gsl::at(x_isolated, 1), 0., momentum_right_ / masses_[1]);
    for (size_t i = 0; i < 2; ++i) {
      gsl::at(boost_vars, i) = get_isolated_vars<requested_boost_tags>(
          *gsl::at(superposed_objects_, i), gsl::at(x_unboosted, i));
    }

    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{std::move(mesh),
                                std::move(inv_jacobian),
                                x,
                                masses_,
                                xcoords_,
                                momentum_left_,
                                momentum_right_,
                                y_offset_,
                                z_offset_,
                                attenuation_parameter_,
                                attenuation_radius_,
                                std::move(x_isolated),
                                std::move(flat_vars),
                                std::move(isolated_vars),
                                std::move(boost_vars)};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  template <typename TagsList, typename... Args>
  tuples::tagged_tuple_from_typelist<TagsList> get_isolated_vars(
      const IsolatedObjectBase& isolated_object, const Args&... args) const {
    return call_with_dynamic_type<tuples::tagged_tuple_from_typelist<TagsList>,
                                  IsolatedObjectClasses>(
        &isolated_object, [&args...](const auto* const derived) {
          return derived->variables(args..., TagsList{});
        });
  }
};

/// \cond
template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
PUP::able::PUP_ID BinaryWithGravitationalWaves<
    IsolatedObjectBase, IsolatedObjectClasses>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Xcts::AnalyticData
