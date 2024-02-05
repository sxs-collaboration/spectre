// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <optional>

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
struct RadiusLeft : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct RadiusRight : db::SimpleTag {
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
struct NormalLR : db::SimpleTag {
  using type = tnsr::I<DataType, 3>;
};
template <typename DataType>
struct MomentumLeft : db::SimpleTag {
  using type = tnsr::I<DataType, 3>;
};
template <typename DataType>
struct MomentumRight : db::SimpleTag {
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
}  // namespace Tags

template <typename DataType>
using WavyBBHVariablesCache = cached_temp_buffer_from_typelist<tmpl::append<
    common_tags<DataType>,
    tmpl::list<
        detail::Tags::RadiusLeft<DataType>, detail::Tags::RadiusRight<DataType>,
        detail::Tags::NormalLeft<DataType>, detail::Tags::NormalRight<DataType>,
        detail::Tags::NormalLR<DataType>, detail::Tags::MomentumLeft<DataType>,
        detail::Tags::MomentumRight<DataType>,
        detail::Tags::RadiativeTerm<DataType>,
        detail::Tags::NearZoneTerm<DataType>,
        detail::Tags::PresentTerm<DataType>, detail::Tags::PastTerm<DataType>,
        detail::Tags::IntegralTerm<DataType>,
        ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
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
struct WavyBBHVariables
    : CommonVariables<DataType, WavyBBHVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = WavyBBHVariablesCache<DataType>;
  using Base = CommonVariables<DataType, WavyBBHVariablesCache<DataType>>;
  using Base::operator();

  WavyBBHVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, 3>& local_x, const double local_mass_left,
      const double local_mass_right, const double local_xcoord_left,
      const double local_xcoord_right, const double local_ymomentum_left,
      const double local_ymomentum_right, const double local_fat_par)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        mass_left(local_mass_left),
        mass_right(local_mass_right),
        xcoord_left(local_xcoord_left),
        xcoord_right(local_xcoord_right),
        ymomentum_left(local_ymomentum_left),
        ymomentum_right(local_ymomentum_right),
        fat_par(local_fat_par) {}

  const tnsr::I<DataType, 3>& x;
  const double mass_left;
  const double mass_right;
  const double xcoord_left;
  const double xcoord_right;
  const double ymomentum_left;
  const double ymomentum_right;
  const double fat_par;
  const double separation = xcoord_right - xcoord_left;

  void operator()(gsl::not_null<Scalar<DataType>*> radius_left,
                  gsl::not_null<Cache*> /*cache*/,
                  detail::Tags::RadiusLeft<DataType> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> radius_right,
                  gsl::not_null<Cache*> /*cache*/,
                  detail::Tags::RadiusRight<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> normal_left,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::NormalLeft<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> normal_right,
                  gsl::not_null<Cache*> cache,
                  detail::Tags::NormalRight<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> normal_LR,
                  gsl::not_null<Cache*> /*cache*/,
                  detail::Tags::NormalLR<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> momentum_left,
                  gsl::not_null<Cache*> /*cache*/,
                  detail::Tags::MomentumLeft<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> momentum_right,
                  gsl::not_null<Cache*> /*cache*/,
                  detail::Tags::MomentumRight<DataType> /*meta*/) const;
  void operator()(const gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
                  const gsl::not_null<Cache*> cache,
                  detail::Tags::RadiativeTerm<DataType> /*meta*/) const;
  void operator()(const gsl::not_null<tnsr::ii<DataType, Dim>*> near_zone_term,
                  const gsl::not_null<Cache*> cache,
                  detail::Tags::NearZoneTerm<DataType> /*meta*/) const;
  void operator()(const gsl::not_null<tnsr::ii<DataType, Dim>*> present_term,
                  const gsl::not_null<Cache*> cache,
                  detail::Tags::PresentTerm<DataType> /*meta*/) const;
  void operator()(const gsl::not_null<tnsr::ii<DataType, Dim>*> past_term,
                  const gsl::not_null<Cache*> cache,
                  detail::Tags::PastTerm<DataType> /*meta*/) const;
  void operator()(const gsl::not_null<tnsr::ii<DataType, Dim>*> integral_term,
                  const gsl::not_null<Cache*> cache,
                  detail::Tags::IntegralTerm<DataType> /*meta*/) const;

  void operator()(
      const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      const gsl::not_null<Cache*> cache,
      Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      const gsl::not_null<Cache*> /*cache*/,
      ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>
          meta) const override;
  void operator()(
      const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      const gsl::not_null<Cache*> /*cache*/,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const override;
  void operator()(
      const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
      const gsl::not_null<Cache*> /*cache*/,
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
      gsl::not_null<Cache*> /*cache*/,
      ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  void operator()(
      const gsl::not_null<Scalar<DataType>*> conformal_energy_density,
      const gsl::not_null<Cache*> /*cache*/,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const;
  void operator()(
      const gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
      const gsl::not_null<Cache*> /*cache*/,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const;
  void operator()(
      const gsl::not_null<tnsr::I<DataType, Dim>*> conformal_momentum_density,
      const gsl::not_null<Cache*> /*cache*/,
      gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
      const;
  void operator()(
      const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
      const gsl::not_null<Cache*> /*cache*/,
      Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      const gsl::not_null<Scalar<DataType>*>
          lapse_times_conformal_factor_minus_one,
      const gsl::not_null<Cache*> /*cache*/,
      Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
      const gsl::not_null<Cache*> /*cache*/,
      Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const;
  void operator()(const gsl::not_null<Scalar<DataType>*> rest_mass_density,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::RestMassDensity<DataType> /*meta*/) const;
  void operator()(const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::SpecificEnthalpy<DataType> /*meta*/) const;
  void operator()(const gsl::not_null<Scalar<DataType>*> pressure,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::Pressure<DataType> /*meta*/) const;
  void operator()(const gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::SpatialVelocity<DataType, 3> /*meta*/) const;
  void operator()(const gsl::not_null<Scalar<DataType>*> lorentz_factor,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::LorentzFactor<DataType> /*meta*/) const;
  void operator()(const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::MagneticField<DataType, 3> /*meta*/) const;

 private:
  void add_radiative_term_PN_of_conformal_metric(
      const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      const gsl::not_null<Cache*> cache) const;
  void add_near_zone_term_to_radiative(
      const gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
      const gsl::not_null<Cache*> cache) const;
  void add_present_term_to_radiative(
      const gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
      const gsl::not_null<Cache*> cache) const;
  void add_past_term_to_radiative(
      const gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
      const gsl::not_null<Cache*> cache) const;
  void add_integral_term_to_radiative(
      const gsl::not_null<tnsr::ii<DataType, Dim>*> radiative_term,
      const gsl::not_null<Cache*> cache) const;
};

}  // namespace detail

/*!
 * \brief   Binary black hole initial data with realistic wave background,
 * constructed in Post-Newtonian approximations.
 *
 * This class implements background data for the XCTS equations describing...
 */
class WavyBBH : public elliptic::analytic_data::Background,
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
  struct YMomentumLeft {
    static constexpr Options::String help =
        "The y-axis-componet of the linear momentum of the left black hole.";
    using type = double;
  };
  struct YMomentumRight {
    static constexpr Options::String help =
        "The y-axis-componet of the linear momentum of the right black hole.";
    using type = double;
  };
  struct FatPar {
    static constexpr Options::String help =
        "The parameter controlling the width of the atenuation function.";
    using type = double;
  };
  using options = tmpl::list<MassLeft, MassRight, XCoordsLeft, XCoordsRight,
                             YMomentumLeft, YMomentumRight, FatPar>;
  static constexpr Options::String help =
      "Binary black hole initial data with realistic wave background, "
      "constructed in Post-Newtonian approximations. ";

  WavyBBH() = default;
  WavyBBH(const WavyBBH&) = delete;
  WavyBBH& operator=(const WavyBBH&) = delete;
  WavyBBH(WavyBBH&&) = default;
  WavyBBH& operator=(WavyBBH&&) = default;
  ~WavyBBH() = default;

  WavyBBH(double mass_left, double mass_right, double xcoord_left,
          double xcoord_right, double ymomentum_left, double ymomentum_right,
          double fat_par, const Options::Context& context = {})
      : mass_left_(mass_left),
        mass_right_(mass_right),
        xcoord_left_(xcoord_left),
        xcoord_right_(xcoord_right),
        ymomentum_left_(ymomentum_left),
        ymomentum_right_(ymomentum_right),
        fat_par_(fat_par) {
    if (mass_left_ <= 0 or mass_right_ <= 0) {
      PARSE_ERROR(context, "'MassLeft' and 'MassRight' need to be positive.");
    }
    if (xcoord_left_ >= xcoord_right_) {
      PARSE_ERROR(context,
                  "'XCoordsLeft' must be smaller than 'XCoordsRight'.");
    }
    if (fat_par_ <= 0) {
      PARSE_ERROR(context, "'FatPar' must be positive.");
    }
    if (ymomentum_left_ * ymomentum_right_ > 0) {
      PARSE_ERROR(
          context,
          "'YMomentumLeft' and 'YMomentumRight' must have opposite signs.");
    }
  }

  explicit WavyBBH(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m),
        elliptic::analytic_data::InitialGuess(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(WavyBBH);

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
    p | fat_par_;
  }

  double mass_left() const { return mass_left_; }
  double mass_right() const { return mass_right_; }
  double xcoord_left() const { return xcoord_left_; }
  double xcoord_right() const { return xcoord_right_; }
  double ymomentum_left() const { return ymomentum_left_; }
  double ymomentum_right() const { return ymomentum_right_; }
  double fat_par() const { return fat_par_; }

 private:
  double mass_left_ = std::numeric_limits<double>::signaling_NaN();
  double mass_right_ = std::numeric_limits<double>::signaling_NaN();
  double xcoord_left_ = std::numeric_limits<double>::signaling_NaN();
  double xcoord_right_ = std::numeric_limits<double>::signaling_NaN();
  double ymomentum_left_ = std::numeric_limits<double>::signaling_NaN();
  double ymomentum_right_ = std::numeric_limits<double>::signaling_NaN();
  double fat_par_ = std::numeric_limits<double>::signaling_NaN();

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<3>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::WavyBBHVariables<DataType>;
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
                                fat_par_};

    return {cache.get_var(computer, RequestedTags{})...};
  }
};

}  // namespace Xcts::AnalyticData
