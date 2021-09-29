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
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "Utilities/Requires.hpp"
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
struct BinaryVariables;

template <typename DataType>
using BinaryVariablesCache = cached_temp_buffer_from_typelist<
    BinaryVariables<DataType>,
    tmpl::push_front<
        common_tags<DataType>,
        Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
        ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        gr::Tags::TraceExtrinsicCurvature<DataType>,
        ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>,
        Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
        ::Tags::deriv<Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
            DataType, 3, Frame::Inertial>,
        gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
        gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
        gr::Tags::Conformal<
            gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>, 0>,
        // For initial guesses
        Tags::ConformalFactor<DataType>,
        Tags::LapseTimesConformalFactor<DataType>,
        Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;

template <typename DataType>
struct BinaryVariables
    : CommonVariables<DataType, BinaryVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = BinaryVariablesCache<DataType>;
  using CommonVariables<DataType, BinaryVariablesCache<DataType>>::operator();

  using superposed_tags = tmpl::list<
      Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
      ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataType>,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
      gr::Tags::Conformal<
          gr::Tags::MomentumDensity<Dim, Frame::Inertial, DataType>, 0>,
      Tags::ConformalFactor<DataType>,
      Tags::LapseTimesConformalFactor<DataType>,
      Tags::ShiftExcess<DataType, Dim, Frame::Inertial>>;

  const tnsr::I<DataVector, Dim>& x;
  const double angular_velocity;
  const std::optional<std::array<double, 2>> falloff_widths;
  const std::array<tnsr::I<DataVector, Dim>, 2> x_isolated;
  const std::array<DataVector, 2> windows;
  const tuples::tagged_tuple_from_typelist<superposed_tags> flat_vars;
  const std::array<tuples::tagged_tuple_from_typelist<superposed_tags>, 2>
      isolated_vars;

  template <typename Tag,
            Requires<tmpl::list_contains_v<superposed_tags, Tag>> = nullptr>
  void operator()(gsl::not_null<typename Tag::type*> superposed_var,
                  gsl::not_null<Cache*> /*cache*/,
                  Tag /*meta*/) const noexcept {
    for (size_t i = 0; i < superposed_var->size(); ++i) {
      (*superposed_var)[i] =
          get<Tag>(flat_vars)[i] +
          windows[0] *
              (get<Tag>(isolated_vars[0])[i] - get<Tag>(flat_vars)[i]) +
          windows[1] * (get<Tag>(isolated_vars[1])[i] - get<Tag>(flat_vars)[i]);
    }
    if constexpr (std::is_same_v<
                      Tag, ::Tags::deriv<Tags::ConformalMetric<DataType, 3,
                                                               Frame::Inertial>,
                                         tmpl::size_t<3>, Frame::Inertial>>) {
      add_deriv_of_window_function(superposed_var);
    }
  }
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const noexcept;

 private:
  void add_deriv_of_window_function(gsl::not_null<tnsr::ijj<DataType, Dim>*>
                                        deriv_conformal_metric) const noexcept;
};
}  // namespace detail

/// \cond
template <typename IsolatedObjectRegistrars, typename Registrars>
class Binary;

namespace Registrars {
template <typename IsolatedObjectRegistrars>
struct Binary {
  template <typename Registrars>
  using f = Xcts::AnalyticData::Binary<IsolatedObjectRegistrars, Registrars>;
};
}  // namespace Registrars
/// \endcond

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
 * \f$w_\alpha\f$ are parameters describing Gaussian falloff-widths. The
 * Gaussian falloffs facilitate that the influence of either of the two objects
 * at the position of the other is strongly damped, and they also avoid
 * logarithmic scaling of the solution at large distances where we would
 * typically employ an inverse-radial coordinate map and asymptotically-flat
 * boundary conditions. The falloff-widths are chosen in terms of the Newtonian
 * Lagrange points of the two objects in \cite Varma2018sqd and
 * \cite Lovelace2008tw, and they are input parameters in this implementation.
 * The falloff can be disabled by passing `std::nullopt` to the constructor, or
 * `None` in the input file.
 *
 * The remaining quantities that this class implements relate to the orbital
 * motion of the two objects. To obtain initial data in "co-rotating"
 * coordinates where the two objects are initially at rest we prescribe the
 * background shift
 *
 * \f{equation} \beta^i_\mathrm{background} = (-\Omega y, \Omega x, 0) \f}
 *
 * where \f$\Omega\f$ is the angular-velocity parameter.
 */
template <typename IsolatedObjectRegistrars,
          typename Registrars = tmpl::list<
              Xcts::AnalyticData::Registrars::Binary<IsolatedObjectRegistrars>>>
class Binary : public ::AnalyticData<3, Registrars> {
 private:
  using Base = ::AnalyticData<3, Registrars>;

 public:
  using IsolatedObjectBase =
      Xcts::Solutions::AnalyticSolution<IsolatedObjectRegistrars>;

  struct XCoords {
    static constexpr Options::String help =
        "The coordinates on the x-axis where the two objects are placed";
    using type = std::array<double, 2>;
  };
  struct ObjectA {
    static constexpr Options::String help =
        "The object placed on the negative x-axis";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };
  struct ObjectB {
    static constexpr Options::String help =
        "The object placed on the positive x-axis";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };
  struct AngularVelocity {
    static constexpr Options::String help = "Orbital angular velocity";
    using type = double;
  };
  struct FalloffWidths {
    static constexpr Options::String help =
        "The widths for the window functions around the two objects, or 'None' "
        "to disable the Gaussian falloff.";
    using type = Options::Auto<std::array<double, 2>, Options::AutoLabel::None>;
  };
  using options =
      tmpl::list<XCoords, ObjectA, ObjectB, AngularVelocity, FalloffWidths>;
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
         std::unique_ptr<IsolatedObjectBase> object_a,
         std::unique_ptr<IsolatedObjectBase> object_b, double angular_velocity,
         std::optional<std::array<double, 2>> falloff_widths) noexcept
      : xcoords_(xcoords),
        superposed_objects_({std::move(object_a), std::move(object_b)}),
        angular_velocity_(angular_velocity),
        falloff_widths_(falloff_widths) {}

  explicit Binary(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Binary);

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    return variables_impl<DataType>(x, std::nullopt, std::nullopt,
                                    tmpl::list<RequestedTags...>{});
  }
  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    return variables_impl<DataVector>(x, mesh, inv_jacobian,
                                      tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) noexcept override {
    Base::pup(p);
    p | xcoords_;
    p | superposed_objects_;
    p | angular_velocity_;
    p | falloff_widths_;
  }

  const std::array<double, 2>& x_coords() const noexcept { return xcoords_; }
  const std::array<std::unique_ptr<IsolatedObjectBase>, 2>& superposed_objects()
      const noexcept {
    return superposed_objects_;
  }
  double angular_velocity() const noexcept { return angular_velocity_; }
  const std::optional<std::array<double, 2>>& falloff_widths() const noexcept {
    return falloff_widths_;
  }

 private:
  std::array<double, 2> xcoords_{};
  std::array<std::unique_ptr<IsolatedObjectBase>, 2> superposed_objects_{};
  Xcts::Solutions::Flatness<> flatness_{};
  double angular_velocity_ = std::numeric_limits<double>::signaling_NaN();
  std::optional<std::array<double, 2>> falloff_widths_{};

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<3>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
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
      gsl::at(isolated_vars, i) =
          gsl::at(superposed_objects_, i)
              ->variables(gsl::at(x_isolated, i), requested_superposed_tags{});
    }
    auto flat_vars = flatness_.variables(x, requested_superposed_tags{});
    typename VarsComputer::Cache cache{
        get_size(*x.begin()),
        VarsComputer{{std::move(mesh), std::move(inv_jacobian)},
                     x,
                     angular_velocity_,
                     falloff_widths_,
                     std::move(x_isolated),
                     std::move(windows),
                     std::move(flat_vars),
                     std::move(isolated_vars)}};
    return {cache.get_var(RequestedTags{})...};
  }
};

/// \cond
template <typename IsolatedObjectRegistrars, typename Registrars>
PUP::able::PUP_ID Binary<IsolatedObjectRegistrars, Registrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace Xcts::AnalyticData
