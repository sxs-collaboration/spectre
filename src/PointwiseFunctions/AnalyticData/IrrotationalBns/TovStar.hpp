// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/IrrotationalBns/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/InitialData/IrrotationalBns.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace IrrotationalBns::InitialData {
namespace tov_detail {

using TovCoordinates = RelativisticEuler::Solutions::TovCoordinates;
using StarRegion = RelativisticEuler::Solutions::tov_detail::StarRegion;

template <typename DataType>
using TovVariablesCache = cached_temp_buffer_from_typelist<tmpl::list<
    IrrotationalBns::Tags::VelocityPotential<DataType>,
    ::Tags::deriv<IrrotationalBns::Tags::VelocityPotential<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>,
    ::Tags::Flux<IrrotationalBns::Tags::VelocityPotential<DataType>,
                 tmpl::size_t<3>, Frame::Inertial>,
    ::Tags::FixedSource<IrrotationalBns::Tags::VelocityPotential<DataType>>,
    gr::Tags::Lapse<DataType>,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::Shift<DataType, 3, Frame::Inertial>,
    ::Tags::deriv<gr::Tags::Shift<DataType, 3>, tmpl::size_t<3>,
                  Frame::Inertial>,
    IrrotationalBns::Tags::RotationalShift<DataType>,
    IrrotationalBns::Tags::DerivLogLapseOverSpecificEnthalpy<DataType>,
    IrrotationalBns::Tags::RotationalShiftStress<DataType>,
    gr::Tags::SpatialMetric<DataType, 3>>>;

template <typename DataType, StarRegion Region = StarRegion::Interior>
struct TovVariables
    : RelativisticEuler::Solutions::tov_detail::TovVariables<DataType, Region> {
  static constexpr size_t Dim = 3;

  using Base =
      RelativisticEuler::Solutions::tov_detail::TovVariables<DataType, Region>;
  using Cache = TovVariablesCache<DataType>;
  using Base::operator();
  using Base::coords;
  using Base::eos;
  using Base::radial_solution;

  const std::array<double, 3> star_center;
  const double euler_enthalpy_constant;
  // Note this is not the angular velocity of the star around its axis, that
  // is zero in this case.
  const double orbital_angular_velocity;
  const tnsr::I<DataType, 3>& x;
  const DataType& radius;
  const RelativisticEuler::Solutions::TovStar& tov_star;

  TovVariables(const tnsr::I<DataType, 3>& local_x,
               const DataType& local_radius,
               const RelativisticEuler::Solutions::TovStar& local_tov_star,
               const EquationsOfState::EquationOfState<true, 1>& local_eos,
               std::array<double, 3> local_star_center,
               const double local_euler_enthalpy_constant,
               const double local_orbital_angular_velocity)
      : Base(local_x, local_radius, local_tov_star.radial_solution(),
             local_eos),
        star_center(std::move(local_star_center)),
        x(local_x),
        radius(local_radius),
        tov_star(local_tov_star),
        euler_enthalpy_constant(local_euler_enthalpy_constant),
        orbital_angular_velocity(local_orbital_angular_velocity){};

  void operator()(
      gsl::not_null<Scalar<DataType>*> velocity_potential,
      gsl::not_null<Cache*> cache,
      IrrotationalBns::Tags::VelocityPotential<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> auxiliary_velocity,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<IrrotationalBns::Tags::VelocityPotential<DataType>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> flux_for_velocity_potential,
      gsl::not_null<Cache*> cache,
      ::Tags::Flux<IrrotationalBns::Tags::VelocityPotential<DataType>,
                   tmpl::size_t<3>, Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> fixed_source,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<
          IrrotationalBns::Tags::VelocityPotential<DataType>> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::i<DataType, 3>*> deriv_lapse,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                                Frame::Inertial> /*meta*/) const;

  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> shift,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Shift<DataType, 3, Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::iJ<DataType, 3>*> deriv_shift,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<gr::Tags::Shift<DataType, 3>, tmpl::size_t<3>,
                                Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> rotational_shift,
      gsl::not_null<Cache*> cache,
      IrrotationalBns::Tags::RotationalShift<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::i<DataType, 3>*>
                      deriv_log_lapse_over_specific_enthalpy,
                  gsl::not_null<Cache*> cache,
                  IrrotationalBns::Tags::DerivLogLapseOverSpecificEnthalpy<
                      DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::II<DataType, 3>*> rotational_shift_stress,
      gsl::not_null<Cache*> cache,
      IrrotationalBns::Tags::RotationalShiftStress<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, 3>*> spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::SpatialMetric<DataType, 3> /*meta*/) const;

 private:
  template <typename Tag>
  typename Tag::type get_tov_var(Tag /*meta*/) const {
    // Possible optimization: Access the cache of the RelEuler::TovStar solution
    // so its intermediate quantities don't have to be re-computed repeatedly
    return get<Tag>(tov_star.variables(
        x, std::numeric_limits<double>::signaling_NaN(), tmpl::list<Tag>{}));
  }
};

}  // namespace tov_detail

/*!
 * \brief TOV solution for irrotational BNS data
 *
 * \see RelativisticEuler::Solutions::TovStar
 * \see gr::Solutions::TovSolution
 * Teh
 */
class TovStar : public elliptic::analytic_data::AnalyticSolution {
 private:
  using RelEulerTovStar = RelativisticEuler::Solutions::TovStar;
  std::array<double, 3> star_center_{};
  double euler_enthalpy_constant_ =
      std::numeric_limits<double>::signaling_NaN();
  // Note this is not the angular velocity of the star around its axis, that
  // is zero in this case.
  double orbital_angular_velocity_ =
      std::numeric_limits<double>::signaling_NaN();
  RelEulerTovStar tov_star_;

 public:
  struct EulerEnthalpyConstant {
    using type = double;
    static constexpr Options::String help =
        "The Euler Enthalpy constant of the star";
  };
  struct StarCenter {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "The location of the center of the star";
  };
  struct OrbitalAngularVelocity {
    using type = double;
    static constexpr Options::String help =
        "The initial angular velocity of the binary orbit";
  };
  using options = tmpl::list<
      EulerEnthalpyConstant, hydro::OptionTags::EquationOfState<true, 1>,
      RelEulerTovStar::Coordinates, StarCenter, OrbitalAngularVelocity>;
  static constexpr Options::String help = {
      "A TovStar to be used as a background"
      "for solving the Irrotational BNS hydro equations"};

  TovStar() = default;
  TovStar(const TovStar&) = default;
  TovStar& operator=(const TovStar&) = default;
  TovStar(TovStar&&) = default;
  TovStar& operator=(TovStar&&) = default;
  ~TovStar() = default;

  // We do not a priori know what the central density will be, a (likely poor)
  // guess we use here is that the central enthalpy is equal to the constant
  // A better approxumation would include some guess for the central lapse and
  // set h = C / alpha
  TovStar(double euler_enthalpy_constant,
          std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>
              equation_of_state,
          const RelativisticEuler::Solutions::TovCoordinates coordinate_system,
          std::array<double, 3> star_center, double orbital_angular_velocity)
      : star_center_(std::move(star_center)),
        euler_enthalpy_constant_(euler_enthalpy_constant),
        orbital_angular_velocity_(orbital_angular_velocity),
        tov_star_(get(equation_of_state->rest_mass_density_from_enthalpy(
                      Scalar<double>(euler_enthalpy_constant))),
                  std::move(equation_of_state), coordinate_system){};

  const EquationsOfState::EquationOfState<true, 1>& equation_of_state() const {
    return tov_star_.equation_of_state();
  }

  const RelativisticEuler::Solutions::TovSolution& radial_solution() const {
    return tov_star_.radial_solution();
  }

  /// \cond
  explicit TovStar(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TovStar);
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone() const {
    return std::make_unique<TovStar>(*this);
  }
  /// \endcond

  template <typename DataType>
  using tags = typename tov_detail::TovVariablesCache<DataType>::tags_list;

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataType>(x, tmpl::list<RequestedTags...>{});
  }

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& /*mesh*/,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataVector>(x, tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | tov_star_;
    p | euler_enthalpy_constant_;
    p | star_center_;
    p | orbital_angular_velocity_;
  }

 private:
  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = tov_detail::TovVariables<DataType>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const DataType radius = get(magnitude(x));
    const VarsComputer computer{x,
                                radius,
                                tov_star_,
                                tov_star_.equation_of_state(),
                                star_center_,
                                euler_enthalpy_constant_,
                                orbital_angular_velocity_};
  }

  friend bool operator==(const TovStar& lhs, const TovStar& rhs) {
    return lhs.tov_star_ == rhs.tov_star_;
  }

};

inline bool operator!=(const TovStar& lhs, const TovStar& rhs) {
  return not(lhs == rhs);
}
}  // namespace IrrotationalBns::InitialData
