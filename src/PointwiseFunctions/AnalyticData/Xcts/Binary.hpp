// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <optional>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Requires.hpp"
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

template <typename DataType>
using BinaryVariablesCache = cached_temp_buffer_from_typelist<tmpl::append<
    common_tags<DataType>,
    tmpl::list<
        ::Tags::deriv<Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
        gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
        gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0>,
        // For initial guesses
        Tags::ConformalFactor<DataType>,
        Tags::LapseTimesConformalFactor<DataType>,
        Tags::ShiftExcess<DataType, 3, Frame::Inertial>>,
    hydro_tags<DataType>>>;

template <typename DataType>
struct BinaryVariables
    : CommonVariables<DataType, BinaryVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = BinaryVariablesCache<DataType>;
  using Base = CommonVariables<DataType, BinaryVariablesCache<DataType>>;
  using Base::operator();

  using superposed_tags = tmpl::append<
      tmpl::list<
          Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
          ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                        tmpl::size_t<Dim>, Frame::Inertial>,
          gr::Tags::TraceExtrinsicCurvature<DataType>,
          ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>,
          gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
          gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
          gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0>,
          Tags::ConformalFactor<DataType>,
          Tags::LapseTimesConformalFactor<DataType>,
          Tags::ShiftExcess<DataType, Dim, Frame::Inertial>>,
      hydro_tags<DataType>>;

  BinaryVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataVector, Dim>& local_x,
      const double local_angular_velocity, const double local_expansion,
      std::optional<std::array<double, 2>> local_falloff_widths,
      std::array<tnsr::I<DataVector, Dim>, 2> local_x_isolated,
      std::array<DataVector, 2> local_windows,
      tuples::tagged_tuple_from_typelist<superposed_tags> local_flat_vars,
      std::array<tuples::tagged_tuple_from_typelist<superposed_tags>, 2>
          local_isolated_vars)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        angular_velocity(local_angular_velocity),
        expansion(local_expansion),
        falloff_widths(std::move(local_falloff_widths)),
        x_isolated(std::move(local_x_isolated)),
        windows(std::move(local_windows)),
        flat_vars(std::move(local_flat_vars)),
        isolated_vars(std::move(local_isolated_vars)) {}

  const tnsr::I<DataVector, Dim>& x;
  const double angular_velocity;
  const double expansion;
  const std::optional<std::array<double, 2>> falloff_widths;
  const std::array<tnsr::I<DataVector, Dim>, 2> x_isolated;
  const std::array<DataVector, 2> windows;
  const tuples::tagged_tuple_from_typelist<superposed_tags> flat_vars;
  const std::array<tuples::tagged_tuple_from_typelist<superposed_tags>, 2>
      isolated_vars;

  template <bool ApplyWindow = true, typename Tag,
            Requires<tmpl::list_contains_v<superposed_tags, Tag>> = nullptr>
  void superposition(gsl::not_null<typename Tag::type*> superposed_var,
                     gsl::not_null<Cache*> /*cache*/, Tag /*meta*/) const {
    for (size_t i = 0; i < superposed_var->size(); ++i) {
      if constexpr (ApplyWindow) {
        (*superposed_var)[i] =
            get<Tag>(flat_vars)[i] +
            windows[0] *
                (get<Tag>(isolated_vars[0])[i] - get<Tag>(flat_vars)[i]) +
            windows[1] *
                (get<Tag>(isolated_vars[1])[i] - get<Tag>(flat_vars)[i]);
      } else {
        (*superposed_var)[i] = get<Tag>(isolated_vars[0])[i] +
                               get<Tag>(isolated_vars[1])[i] -
                               get<Tag>(flat_vars)[i];
      }
    }
  }

  void operator()(
      const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      const gsl::not_null<Cache*> cache,
      Tags::ConformalMetric<DataType, Dim, Frame::Inertial> meta)
      const override {
    superposition(conformal_metric, cache, meta);
  }
  void operator()(
      const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      const gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>
          meta) const override {
    superposition(deriv_conformal_metric, cache, meta);
    add_deriv_of_window_function(deriv_conformal_metric);
  }
  void operator()(
      const gsl::not_null<Scalar<DataType>*> extrinsic_curvature_trace,
      const gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> meta) const override {
    superposition(extrinsic_curvature_trace, cache, meta);
  }
  void operator()(
      const gsl::not_null<Scalar<DataType>*> dt_extrinsic_curvature_trace,
      const gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> meta)
      const override {
    superposition(dt_extrinsic_curvature_trace, cache, meta);
  }
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const override;
  void operator()(
      const gsl::not_null<Scalar<DataType>*> conformal_energy_density,
      const gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> meta) const {
    superposition<false>(conformal_energy_density, cache, meta);
  }
  void operator()(
      const gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
      const gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> meta) const {
    superposition<false>(conformal_stress_trace, cache, meta);
  }
  void operator()(
      const gsl::not_null<tnsr::I<DataType, Dim>*> conformal_momentum_density,
      const gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> meta)
      const {
    superposition<false>(conformal_momentum_density, cache, meta);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> conformal_factor,
                  const gsl::not_null<Cache*> cache,
                  Tags::ConformalFactor<DataType> meta) const {
    superposition(conformal_factor, cache, meta);
  }
  void operator()(
      const gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
      const gsl::not_null<Cache*> cache,
      Tags::LapseTimesConformalFactor<DataType> meta) const {
    superposition(lapse_times_conformal_factor, cache, meta);
  }
  void operator()(
      const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
      const gsl::not_null<Cache*> cache,
      Tags::ShiftExcess<DataType, Dim, Frame::Inertial> meta) const {
    superposition(shift_excess, cache, meta);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> rest_mass_density,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::RestMassDensity<DataType> meta) const {
    superposition<false>(rest_mass_density, cache, meta);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::SpecificEnthalpy<DataType> meta) const {
    superposition<false>(specific_enthalpy, cache, meta);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> pressure,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::Pressure<DataType> meta) const {
    superposition<false>(pressure, cache, meta);
  }
  void operator()(const gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::SpatialVelocity<DataType, 3> meta) const {
    superposition<false>(spatial_velocity, cache, meta);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> lorentz_factor,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::LorentzFactor<DataType> meta) const {
    superposition<false>(lorentz_factor, cache, meta);
  }
  void operator()(const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
                  const gsl::not_null<Cache*> cache,
                  hydro::Tags::MagneticField<DataType, 3> meta) const {
    superposition<false>(magnetic_field, cache, meta);
  }

 private:
  void add_deriv_of_window_function(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric) const;
};
}  // namespace detail

/*!
 * \brief Binary compact-object data in general relativity, constructed from
 * superpositions of two isolated objects.
 *
 * This class implements background data for the XCTS equations describing two
 * objects in a quasi-equilibrium orbit, i.e. with \f$\bar{u}=0\f$ and
 * \f$\partial_t K=0\f$. Both objects can be chosen from the list of
 * `IsolatedObjectRegistrars`, e.g. they can be black-hole or neutron-star
 * solutions in different coordinates. Most quantities are constructed by
 * superposing the two isolated solutions (see e.g. Eq. (8-9) in
 * \cite Varma2018sqd or Eq. (45-46) in \cite Lovelace2008tw):
 *
 * \f{align}
 * \bar{\gamma}_{ij} &= f_{ij} + \sum_{\alpha=1}^2
 * e^{-r_\alpha^2 / w_\alpha^2}\left(\gamma^\alpha_{ij} - f_{ij}\right) \\
 * K &= \sum_{\alpha=1}^2 e^{-r_\alpha^2 / w_\alpha^2}K^\alpha
 * \f}
 *
 * where \f$\gamma^\alpha_{ij}\f$ and \f$K^\alpha\f$ denote the spatial metric
 * and extrinsic-curvature trace of the two individual solutions, \f$r_\alpha\f$
 * is the Euclidean coordinate-distance from the center of each object and
 * \f$w_\alpha\f$ are parameters describing the falloff widths of Gaussian
 * window functions. The window functions
 * facilitate that the influence of either of the two objects
 * at the position of the other is strongly damped, and they also avoid
 * logarithmic scaling of the solution at large distances where we would
 * typically employ an inverse-radial coordinate map and asymptotically-flat
 * boundary conditions. The falloff-widths are chosen in terms of the Newtonian
 * Lagrange points of the two objects in \cite Varma2018sqd and
 * \cite Lovelace2008tw, and they are input parameters in this implementation.
 * The falloff can be disabled by passing `std::nullopt` to the constructor, or
 * `None` in the input file.
 *
 * \par Matter sources
 * Matter sources are superposed without the window functions. The analytic
 * matter sources are of
 * limited use anyway, because in a binary setting they don't take the
 * gravitational influence of the other body into account. Therefore, the matter
 * sources should typically be solved-for alongside the gravity sector to impose
 * conditions such as hydrostatic equilibrium. For scenarios where we just want
 * to superpose the isolated matter solutions and compute the resulting gravity,
 * the matter sources are simply added.
 *
 * \par Orbital motion
 * The remaining quantities that this class implements relate to the orbital
 * motion of the two objects. To obtain initial data in "co-rotating"
 * coordinates where the two objects are initially at rest we prescribe the
 * background shift
 *
 * \f{equation} \beta^i_\mathrm{background} = (-\Omega y, \Omega x, 0) +
 * \dot{a}_0 x^i \f}
 *
 * where \f$\Omega\f$ is the angular-velocity parameter and \f$\dot{a}_0\f$
 * is an expansion parameter. Both control the eccentricity of the orbit.
 */
template <typename IsolatedObjectBase, typename IsolatedObjectClasses>
class Binary : public elliptic::analytic_data::Background,
               public elliptic::analytic_data::InitialGuess {
 public:
  struct XCoords {
    static constexpr Options::String help =
        "The coordinates on the x-axis where the two objects are placed";
    using type = std::array<double, 2>;
  };
  struct ObjectLeft {
    static constexpr Options::String help =
        "The object placed on the negative x-axis";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };
  struct ObjectRight {
    static constexpr Options::String help =
        "The object placed on the positive x-axis";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };
  struct AngularVelocity {
    static constexpr Options::String help =
        "Orbital angular velocity 'Omega0' about the z-axis. Added to the "
        "background shift as a term 'Omega0 x r'.";
    using type = double;
  };
  struct Expansion {
    static constexpr Options::String help =
        "The expansion parameter 'adot0', which is a radial velocity over "
        "radius. Added to the background shift as a term 'adot0 r^i'";
    using type = double;
  };
  struct FalloffWidths {
    static constexpr Options::String help =
        "The widths for the window functions around the two objects, or 'None' "
        "to disable the Gaussian falloff.";
    using type = Options::Auto<std::array<double, 2>, Options::AutoLabel::None>;
  };
  using options = tmpl::list<XCoords, ObjectLeft, ObjectRight, AngularVelocity,
                             Expansion, FalloffWidths>;
  static constexpr Options::String help =
      "Binary compact-object data in general relativity, constructed from "
      "superpositions of two isolated objects.";

  Binary() = default;
  Binary(const Binary&) = delete;
  Binary& operator=(const Binary&) = delete;
  Binary(Binary&&) = default;
  Binary& operator=(Binary&&) = default;
  ~Binary() = default;

  Binary(std::array<double, 2> xcoords,
         std::unique_ptr<IsolatedObjectBase> object_left,
         std::unique_ptr<IsolatedObjectBase> object_right,
         double angular_velocity, const double expansion,
         std::optional<std::array<double, 2>> falloff_widths,
         const Options::Context& context = {})
      : xcoords_(xcoords),
        superposed_objects_({std::move(object_left), std::move(object_right)}),
        angular_velocity_(angular_velocity),
        expansion_(expansion),
        falloff_widths_(falloff_widths) {
    if (xcoords_[0] >= xcoords_[1]) {
      PARSE_ERROR(context, "Specify 'XCoords' ascending from left to right.");
    }
  }

  explicit Binary(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m),
        elliptic::analytic_data::InitialGuess(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Binary);

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
    p | superposed_objects_;
    p | angular_velocity_;
    p | expansion_;
    p | falloff_widths_;
  }

  /// Coordinates of the objects, ascending left to right
  const std::array<double, 2>& x_coords() const { return xcoords_; }
  /// The two objects. First entry is the left object, second entry is the right
  /// object.
  const std::array<std::unique_ptr<IsolatedObjectBase>, 2>& superposed_objects()
      const {
    return superposed_objects_;
  }
  double angular_velocity() const { return angular_velocity_; }
  double expansion() const { return expansion_; }
  const std::optional<std::array<double, 2>>& falloff_widths() const {
    return falloff_widths_;
  }

 private:
  std::array<double, 2> xcoords_{};
  std::array<std::unique_ptr<IsolatedObjectBase>, 2> superposed_objects_{};
  Xcts::Solutions::Flatness flatness_{};
  double angular_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double expansion_ = std::numeric_limits<double>::signaling_NaN();
  std::optional<std::array<double, 2>> falloff_widths_{};

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<3>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    std::array<tnsr::I<DataVector, 3>, 2> x_isolated{{x, x}};
    std::array<DataVector, 2> euclidean_distance{};
    std::array<DataVector, 2> windows{};
    // Possible optimization: Only retrieve those superposed tags from the
    // isolated solutions that are actually needed. This needs some dependency
    // logic, because some of the non-superposed tags depend on superposed tags.
    using VarsComputer = detail::BinaryVariables<DataType>;
    using requested_superposed_tags = typename VarsComputer::superposed_tags;
    std::array<tuples::tagged_tuple_from_typelist<requested_superposed_tags>, 2>
        isolated_vars;
    for (size_t i = 0; i < 2; ++i) {
      get<0>(gsl::at(x_isolated, i)) -= gsl::at(xcoords_, i);
      gsl::at(euclidean_distance, i) = get(magnitude(gsl::at(x_isolated, i)));
      if (falloff_widths_.has_value()) {
        gsl::at(windows, i) = exp(-square(gsl::at(euclidean_distance, i)) /
                                  square(gsl::at(*falloff_widths_, i)));
      } else {
        gsl::at(windows, i) = make_with_value<DataVector>(x, 1.);
      }
      gsl::at(isolated_vars, i) = get_isolated_vars<requested_superposed_tags>(
          *gsl::at(superposed_objects_, i), gsl::at(x_isolated, i));
    }
    auto flat_vars = flatness_.variables(x, requested_superposed_tags{});
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{std::move(mesh),
                                std::move(inv_jacobian),
                                x,
                                angular_velocity_,
                                expansion_,
                                falloff_widths_,
                                std::move(x_isolated),
                                std::move(windows),
                                std::move(flat_vars),
                                std::move(isolated_vars)};
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
PUP::able::PUP_ID Binary<IsolatedObjectBase, IsolatedObjectClasses>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace Xcts::AnalyticData
