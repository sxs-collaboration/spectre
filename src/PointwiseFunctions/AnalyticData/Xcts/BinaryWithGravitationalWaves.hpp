// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <optional>
#include <vector>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::AnalyticData {

namespace detail {

namespace Tags {
template <typename DataType>
struct DistanceLeft : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct OneOverDistanceLeft : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct DistanceRight : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct OneOverDistanceRight : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct NormalLeft : db::SimpleTag {
  using type = tnsr::I<DataType, 3>;
};
template <typename DataType>
struct NormalRight : db::SimpleTag {
  using type = tnsr::I<DataType, 3>;
};
template <typename DataType>
struct RadiativeTerm : db::SimpleTag {
  using type = tnsr::ii<DataType, 3>;
};
template <typename DataType>
struct NearZoneTerm : db::SimpleTag {
  using type = tnsr::ii<DataType, 3>;
};
template <typename DataType>
struct PresentTerm : db::SimpleTag {
  using type = tnsr::ii<DataType, 3>;
};
template <typename DataType>
struct PastTerm : db::SimpleTag {
  using type = tnsr::ii<DataType, 3>;
};
template <typename DataType>
struct IntegralTerm : db::SimpleTag {
  using type = tnsr::ii<DataType, 3>;
};
template <typename DataType>
struct PostNewtonianConjugateMomentum3 : db::SimpleTag {
  using type = tnsr::ii<DataType, 3>;
};
template <typename DataType>
struct PostNewtonianExtrinsicCurvature : db::SimpleTag {
  using type = tnsr::ii<DataType, 3>;
};
template <typename DataType>
struct RetardedTimeLeft : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct RetardedTimeRight : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct RootFinderBracketTimeLower : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct RootFinderBracketTimeUpper : db::SimpleTag {
  using type = Scalar<DataType>;
};
}  // namespace Tags

template <typename DataType>
using BinaryWithGravitationalWavesVariablesCache =
    cached_temp_buffer_from_typelist<tmpl::append<
        common_tags<DataType>,
        tmpl::list<
            ::Tags::deriv<detail::Tags::OneOverDistanceLeft<DataType>,
                          tmpl::size_t<3>, Frame::Inertial>,
            ::Tags::deriv<detail::Tags::OneOverDistanceRight<DataType>,
                          tmpl::size_t<3>, Frame::Inertial>,
            ::Tags::deriv<
                ::Tags::deriv<
                    ::Tags::deriv<detail::Tags::DistanceLeft<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
                tmpl::size_t<3>, Frame::Inertial>,
            ::Tags::deriv<
                ::Tags::deriv<
                    ::Tags::deriv<detail::Tags::DistanceRight<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
                tmpl::size_t<3>, Frame::Inertial>,
            detail::Tags::DistanceLeft<DataType>,
            detail::Tags::DistanceRight<DataType>,
            detail::Tags::NormalLeft<DataType>,
            detail::Tags::NormalRight<DataType>,
            detail::Tags::RadiativeTerm<DataType>,
            detail::Tags::NearZoneTerm<DataType>,
            detail::Tags::PresentTerm<DataType>,
            detail::Tags::PastTerm<DataType>,
            detail::Tags::IntegralTerm<DataType>,
            detail::Tags::PostNewtonianConjugateMomentum3<DataType>,
            detail::Tags::PostNewtonianExtrinsicCurvature<DataType>,
            detail::Tags::RetardedTimeLeft<DataType>,
            detail::Tags::RetardedTimeRight<DataType>,
            detail::Tags::RootFinderBracketTimeLower<DataType>,
            detail::Tags::RootFinderBracketTimeUpper<DataType>,
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

  BinaryWithGravitationalWavesVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, 3>& local_x, const double local_mass_left,
      const double local_mass_right, const double local_xcoord_left,
      const double local_xcoord_right, const double local_ymomentum_left,
      const double local_ymomentum_right,
      const double local_attenuation_parameter,
      const std::array<std::vector<double>, 3>& local_past_position_left,
      const std::array<std::vector<double>, 3>& local_past_position_right,
      const std::array<std::vector<double>, 3>& local_past_dt_position_left,
      const std::array<std::vector<double>, 3>& local_past_dt_position_right,
      const std::array<std::vector<double>, 3>& local_past_momentum_left,
      const std::array<std::vector<double>, 3>& local_past_momentum_right,
      const std::array<std::vector<double>, 3>& local_past_dt_momentum_left,
      const std::array<std::vector<double>, 3>& local_past_dt_momentum_right,
      const std::vector<double>& local_past_time)
      : Base(local_mesh, local_inv_jacobian),
        mesh(std::move(local_mesh)),
        inv_jacobian(std::move(local_inv_jacobian)),
        x(local_x),
        mass_left(local_mass_left),
        mass_right(local_mass_right),
        xcoord_left(local_xcoord_left),
        xcoord_right(local_xcoord_right),
        ymomentum_left(local_ymomentum_left),
        ymomentum_right(local_ymomentum_right),
        attenuation_parameter(local_attenuation_parameter),
        past_position_left(local_past_position_left),
        past_position_right(local_past_position_right),
        past_dt_position_left(local_past_dt_position_left),
        past_dt_position_right(local_past_dt_position_right),
        past_momentum_left(local_past_momentum_left),
        past_momentum_right(local_past_momentum_right),
        past_dt_momentum_left(local_past_dt_momentum_left),
        past_dt_momentum_right(local_past_dt_momentum_right),
        past_time(local_past_time) {
    interpolate_past_history();
  }

  std::optional<std::reference_wrapper<const Mesh<Dim>>> mesh;
  std::optional<std::reference_wrapper<const InverseJacobian<
      DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
      inv_jacobian;

  const tnsr::I<DataType, 3>& x;
  const double mass_left;
  const double mass_right;
  const double xcoord_left;
  const double xcoord_right;
  const double ymomentum_left;
  const double ymomentum_right;
  const double attenuation_parameter;
  const double separation = xcoord_right - xcoord_left;
  const std::array<double, 3> normal_lr{{-1., 0., 0.}};
  const std::array<double, 3> momentum_left{{0., ymomentum_left, 0.}};
  const std::array<double, 3> momentum_right{{0., ymomentum_right, 0.}};

  const std::array<std::vector<double>, 3>& past_position_left{};
  const std::array<std::vector<double>, 3>& past_position_right{};
  const std::array<std::vector<double>, 3>& past_dt_position_left{};
  const std::array<std::vector<double>, 3>& past_dt_position_right{};
  const std::array<std::vector<double>, 3>& past_momentum_left{};
  const std::array<std::vector<double>, 3>& past_momentum_right{};
  const std::array<std::vector<double>, 3>& past_dt_momentum_left{};
  const std::array<std::vector<double>, 3>& past_dt_momentum_right{};
  const std::vector<double>& past_time{};

  std::array<std::function<double(double)>, 3> interpolation_position_left{};
  std::array<std::function<double(double)>, 3> interpolation_position_right{};
  std::array<std::function<double(double)>, 3> interpolation_momentum_left{};
  std::array<std::function<double(double)>, 3> interpolation_momentum_right{};

  void operator()(gsl::not_null<Scalar<DataType>*> distance_left,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::DistanceLeft<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> distance_right,
                  gsl::not_null<Cache*> /*cache*/,
                  detail::Tags::DistanceRight<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> deriv_one_over_distance_left,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<detail::Tags::OneOverDistanceLeft<DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> deriv_one_over_distance_right,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<detail::Tags::OneOverDistanceRight<DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ijk<DataType, Dim>*> deriv_3_distance_left,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<
          ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceLeft<DataType>,
                                      tmpl::size_t<Dim>, Frame::Inertial>,
                        tmpl::size_t<Dim>, Frame::Inertial>,
          tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ijk<DataType, Dim>*> deriv_3_distance_right,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<
          ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceRight<DataType>,
                                      tmpl::size_t<Dim>, Frame::Inertial>,
                        tmpl::size_t<Dim>, Frame::Inertial>,
          tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, Dim>*> normal_left,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::NormalLeft<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, Dim>*> normal_right,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::NormalRight<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::RadiativeTerm<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> near_zone_term,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::NearZoneTerm<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> present_term,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::PresentTerm<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> past_term,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::PastTerm<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> integral_term,
                  gsl::not_null<Cache*> /*cache*/,
                  detail::Tags::IntegralTerm<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> pn_conjugate_momentum3,
      gsl::not_null<Cache*> cache,
      detail::Tags::PostNewtonianConjugateMomentum3<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> pn_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      detail::Tags::PostNewtonianExtrinsicCurvature<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> retarded_time_left,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::RetardedTimeLeft<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> retarded_time_right,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::RetardedTimeRight<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> rootfinder_bracket_time_lower,
      gsl::not_null<Cache*> /*cache*/,
      detail::Tags::RootFinderBracketTimeLower<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> rootfinder_bracket_time_upper,
      gsl::not_null<Cache*> /*cache*/,
      detail::Tags::RootFinderBracketTimeUpper<DataType> /*meta*/) const;

  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      gsl::not_null<Cache*> cache,
      Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      gsl::not_null<Cache*> /*cache*/,
      ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>
          meta) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      gsl::not_null<Cache*> /*cache*/,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Cache*> /*cache*/,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> /*cache*/,
      Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> /*cache*/,
                  Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> conformal_energy_density,
      gsl::not_null<Cache*> /*cache*/,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
      gsl::not_null<Cache*> /*cache*/,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> conformal_momentum_density,
      gsl::not_null<Cache*> /*cache*/,
      gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
      const;
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
                  gsl::not_null<Cache*> /*cache*/,
                  Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<Cache*> /*cache*/,
      Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
      gsl::not_null<Cache*> /*cache*/,
      Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> rest_mass_density,
                  gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::RestMassDensity<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> specific_enthalpy,
                  gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::SpecificEnthalpy<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> pressure,
                  gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::Pressure<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
                  gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::SpatialVelocity<DataType, 3> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> lorentz_factor,
                  gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::LorentzFactor<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
                  gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::MagneticField<DataType, 3> /*meta*/) const;

 private:
  void add_radiative_term_PN_of_conformal_metric(
      gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      gsl::not_null<Cache*> cache) const;
  void add_near_zone_term_to_radiative(
      gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
      gsl::not_null<Cache*> cache) const;
  void add_present_term_to_radiative(
      gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
      gsl::not_null<Cache*> cache) const;
  void add_past_term_to_radiative(
      gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
      gsl::not_null<Cache*> cache) const;
  void add_integral_term_to_radiative(
      gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
      gsl::not_null<Cache*> cache) const;
  Scalar<DataType> this_dot_product(const tnsr::I<DataType, 3>& a,
                                    const std::array<double, 3>& b) const;
  Scalar<DataType> this_dot_product(const std::array<double, 3>& a,
                                    const tnsr::I<DataType, 3>& b) const;
  double max_time_interpolator = std::numeric_limits<double>::signaling_NaN();
  void interpolate_past_history();
  DataType find_retarded_time_left(gsl::not_null<Cache*> cache) const;
  DataType find_retarded_time_right(gsl::not_null<Cache*> cache) const;
  Scalar<DataType> get_past_distance_left(DataType t) const;
  Scalar<DataType> get_past_distance_right(DataType t) const;
  Scalar<DataType> get_past_separation(DataType t) const;
  tnsr::I<DataType, 3> get_past_momentum_left(DataType t) const;
  tnsr::I<DataType, 3> get_past_momentum_right(DataType t) const;
  tnsr::I<DataType, 3> get_past_normal_left(DataType t) const;
  tnsr::I<DataType, 3> get_past_normal_right(DataType t) const;
  tnsr::I<DataType, 3> get_past_normal_lr(DataType t) const;
};

}  // namespace detail

/*!
 * \brief   Binary black hole initial data with realistic wave background,
 * constructed in Post-Newtonian approximations.
 *
 * The main goal of this implementation is to improve the extracted
 * wave forms, for example, by minimizing junk radiation.
 * The data is only valid for black holes without spin. Even so, there is some
 * work done to describe such systems that could later be implemented. The data
 * is constructed from Post-Newtonian expansions for the inspiral phase, in
 * orders of \f$\epsilon = 1/c\f$, in \cite Jaranowski1997ky. In ADMTT gauge it
 * is possible to get the 3-metric as \f$\gamma^{PN}_{ij} = \psi^{4}_{PN}
 * \delta_{ij} + h^{TT}_{ij}\f$ where \f$h^{TT}_{ij}\f$ is the radiative part
 * and the non-radiative Post-Newtonian conformal factor is given by
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
 * n_b^j+32(\frac{2}{s_{a b}^2}-\frac{1}{r_{a b}^2}) n_a^{(i} n_{a b}^{j)} \\
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
 * \frac{\vec{p_2}}{\sqrt{m_2}}\Bigr] + H^{TT\ 1}_{ij} \Bigr[ \sqrt{\frac{m_1
 * m_2}{2 r_{12}}}  \hat{n_{12}} \Bigr] + H^{TT\ 2}_{ij} \Bigr[ \sqrt{\frac{m_1
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
 * u^2] n_a^i n_a^j + 12 (\vec{u} \cdot \hat{n}_a) u^{(i}n_a^{j)}\Bigr\}_t, \f}
 *
 * \f{equation}{
 * H^{TT\ a}_{ij} [ \vec{u} ; t^{r}_a] = -\frac{1}{r_a(t^{r}_a)} \Bigr\{ [-2u^2
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
 * \warning The integral term, equation \f$\eqref{eq:integral_term}\f$, is not
 * implemented yet. Instead this term is set to zero.
 *
 * With this the whole spatial metric is computed up to \f$4PN\f$ order and the
 * radiative term agrees well with quadrupole predictions. In \cite Tichy2002ec,
 * the extrinsic curvature is given up to \f$5PN\f$ order by
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
 * \warning The class is still being worked on. The Solver was not tested yet,
 * for now we still see a very slow convergence.
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
class BinaryWithGravitationalWaves
    : public elliptic::analytic_data::Background,
      public elliptic::analytic_data::InitialGuess {
 public:
  struct MassLeft {
    static constexpr Options::String help = "The mass of the left black hole.";
    using type = double;
  };
  struct MassRight {
    static constexpr Options::String help = "The mass of the right black hole.";
    using type = double;
  };
  struct XCoordsLeft {
    static constexpr Options::String help =
        "The coordinates on the x-axis of the left black hole.";
    using type = double;
  };
  struct XCoordsRight {
    static constexpr Options::String help =
        "The coordinates on the x-axis of the right black hole.";
    using type = double;
  };
  struct AttenuationParameter {
    static constexpr Options::String help =
        "The parameter controlling the width of the attenuation function.";
    using type = double;
  };
  struct OuterRadius {
    static constexpr Options::String help =
        "The radius of the outer boundary of the computational domain.";
    using type = double;
  };
  struct WriteEvolutionOption {
    static constexpr Options::String help =
        "Option to write the evolution of the past history to a file.";
    using type = bool;
  };
  using options =
      tmpl::list<MassLeft, MassRight, XCoordsLeft, XCoordsRight,
                 AttenuationParameter, OuterRadius, WriteEvolutionOption>;
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

  BinaryWithGravitationalWaves(double mass_left, double mass_right,
                               double xcoord_left, double xcoord_right,
                               double attenuation_parameter,
                               double outer_radius, bool write_evolution_option,
                               const Options::Context& context = {})
      : mass_left_(mass_left),
        mass_right_(mass_right),
        xcoord_left_(xcoord_left),
        xcoord_right_(xcoord_right),
        attenuation_parameter_(attenuation_parameter),
        outer_radius_(outer_radius),
        write_evolution_option_(write_evolution_option) {
    if (mass_left_ < 0. or mass_right_ < 0.) {
      PARSE_ERROR(context, "'MassLeft' and 'MassRight' need to be positive.");
    }
    if (xcoord_left_ >= xcoord_right_) {
      PARSE_ERROR(context,
                  "'XCoordsLeft' must be smaller than 'XCoordsRight'.");
    }
    if (attenuation_parameter_ < 0.) {
      PARSE_ERROR(context, "'AttenuationParameter' must be positive.");
    }
    if (outer_radius_ <= 0.) {
      PARSE_ERROR(context, "'OuterRadius' must be positive.");
    }

    // Set the past history data
    initialize();
    reserve_vector_capacity();
    integrate_hamiltonian_system();
    write_evolution_to_file();
    reverse_vector();
  }

  explicit BinaryWithGravitationalWaves(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m),
        elliptic::analytic_data::InitialGuess(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BinaryWithGravitationalWaves);

  template <typename DataType>
  using tags = typename detail::BinaryWithGravitationalWavesVariablesCache<
      DataType>::tags_list;

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
    p | mass_left_;
    p | mass_right_;
    p | xcoord_left_;
    p | xcoord_right_;
    p | ymomentum_left_;
    p | ymomentum_right_;
    p | attenuation_parameter_;
    p | write_evolution_option_;
    p | outer_radius_;
    for (auto& vec : past_position_left_) {
      p | vec;
    }
    for (auto& vec : past_position_right_) {
      p | vec;
    }
    for (auto& vec : past_dt_position_left_) {
      p | vec;
    }
    for (auto& vec : past_dt_position_right_) {
      p | vec;
    }
    for (auto& vec : past_momentum_left_) {
      p | vec;
    }
    for (auto& vec : past_momentum_right_) {
      p | vec;
    }
    for (auto& vec : past_dt_momentum_left_) {
      p | vec;
    }
    for (auto& vec : past_dt_momentum_right_) {
      p | vec;
    }
    p | past_time_;
  }

  double mass_left() const { return mass_left_; }
  double mass_right() const { return mass_right_; }
  double xcoord_left() const { return xcoord_left_; }
  double xcoord_right() const { return xcoord_right_; }
  double attenuation_parameter() const { return attenuation_parameter_; }
  double outer_radius() const { return outer_radius_; }
  bool write_evolution_option() const { return write_evolution_option_; }

 private:
  double mass_left_ = std::numeric_limits<double>::signaling_NaN();
  double mass_right_ = std::numeric_limits<double>::signaling_NaN();
  double xcoord_left_ = std::numeric_limits<double>::signaling_NaN();
  double xcoord_right_ = std::numeric_limits<double>::signaling_NaN();
  double ymomentum_left_ = std::numeric_limits<double>::signaling_NaN();
  double ymomentum_right_ = std::numeric_limits<double>::signaling_NaN();
  double attenuation_parameter_ = std::numeric_limits<double>::signaling_NaN();
  double outer_radius_ = std::numeric_limits<double>::signaling_NaN();
  bool write_evolution_option_ = true;

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<3>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer =
        detail::BinaryWithGravitationalWavesVariables<DataType>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{std::move(mesh),
                                std::move(inv_jacobian),
                                x,
                                mass_left_,
                                mass_right_,
                                xcoord_left_,
                                xcoord_right_,
                                ymomentum_left_,
                                ymomentum_right_,
                                attenuation_parameter_,
                                past_position_left_,
                                past_position_right_,
                                past_dt_position_left_,
                                past_dt_position_right_,
                                past_momentum_left_,
                                past_momentum_right_,
                                past_dt_momentum_left_,
                                past_dt_momentum_right_,
                                past_time_};

    return {cache.get_var(computer, RequestedTags{})...};
  }

  // Implementation of the past history evolution
  double total_mass = 0.;
  double reduced_mass = 0.;
  double reduced_mass_over_total_mass = 0.;
  std::array<double, 3> initial_state_position{};
  std::array<double, 3> initial_state_momentum{};

  std::array<std::vector<double>, 3> past_position_left_{};
  std::array<std::vector<double>, 3> past_position_right_{};
  std::array<std::vector<double>, 3> past_dt_position_left_{};
  std::array<std::vector<double>, 3> past_dt_position_right_{};
  std::array<std::vector<double>, 3> past_momentum_left_{};
  std::array<std::vector<double>, 3> past_momentum_right_{};
  std::array<std::vector<double>, 3> past_dt_momentum_left_{};
  std::array<std::vector<double>, 3> past_dt_momentum_right_{};
  std::vector<double> past_time_{};
  void reserve_vector_capacity();
  void reverse_vector();

  double initial_time = 0.;
  double final_time = 0.;
  double time_step = 0.;
  size_t number_of_steps = 0;

  void initialize();

  using state_type = std::array<double, 6>;

  // Hamiltonian system
  void hamiltonian_system(const state_type& x, state_type& dpdt) const;

  // Observer: store state in vectors
  void observer_vector(const state_type& x, double t);

  // Integration of the Hamiltonian system
  void integrate_hamiltonian_system();
  void write_evolution_to_file() const;
};

}  // namespace Xcts::AnalyticData
