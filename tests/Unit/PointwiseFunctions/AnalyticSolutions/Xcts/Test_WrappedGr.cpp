// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/CcsnCollapse.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Poloidal.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/RotatingStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/PrimitiveSources.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename T>
using Energy = gr::Tags::Conformal<gr::Tags::EnergyDensity<T>, 0>;
template <typename T>
using Momentum = gr::Tags::Conformal<gr::Tags::MomentumDensity<T, 3>, 0>;
template <typename T>
using Trace = gr::Tags::Conformal<gr::Tags::StressTrace<T>, 0>;
template <typename T>
using Enthalpy = hydro::Tags::SpecificEnthalpy<T>;
template <typename T>
using Lorentz = hydro::Tags::LorentzFactor<T>;

std::string fixture(const std::string& relative) {
  return unit_test_src_path() + "/PointwiseFunctions/" + relative;
}

template <typename Provider, typename T, typename Tags>
auto provider_variables(const Provider& provider, const tnsr::I<T, 3>& x,
                        Tags tags) {
  if constexpr (is_analytic_solution_v<Provider>) {
    return provider.variables(x, 0., tags);
  } else {
    return provider.variables(x, tags);
  }
}

template <typename Provider, typename T>
void check_sources(const Xcts::Solutions::WrappedGr<Provider, true>& wrapper,
                   const tnsr::I<T, 3>& x) {
  using hydro::Tags::MagneticField;
  using hydro::Tags::Pressure;
  using hydro::Tags::RestMassDensity;
  using hydro::Tags::SpatialVelocity;
  using hydro::Tags::SpecificInternalEnergy;
  const auto all = wrapper.variables(
      x,
      tmpl::list<Energy<T>, Momentum<T>, Trace<T>, Enthalpy<T>, Lorentz<T>>{});
  CHECK(get<Energy<T>>(all) ==
        get<Energy<T>>(wrapper.variables(x, tmpl::list<Energy<T>>{})));
  CHECK(get<Momentum<T>>(all) ==
        get<Momentum<T>>(wrapper.variables(x, tmpl::list<Momentum<T>>{})));
  CHECK(get<Trace<T>>(all) ==
        get<Trace<T>>(wrapper.variables(x, tmpl::list<Trace<T>>{})));
  const auto native = provider_variables(
      wrapper.gr_solution(), x,
      tmpl::list<RestMassDensity<T>, SpecificInternalEnergy<T>, Pressure<T>,
                 SpatialVelocity<T, 3>, MagneticField<T, 3>, Enthalpy<T>,
                 Lorentz<T>, gr::Tags::SpatialMetric<T, 3>>{});
  CHECK(get<Enthalpy<T>>(all) == get<Enthalpy<T>>(native));
  CHECK(get<Lorentz<T>>(all) == get<Lorentz<T>>(native));
  std::array<T, 3> velocity{};
  std::array<T, 3> magnetic_field{};
  std::array<std::array<T, 3>, 3> metric{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(velocity, i) = get<SpatialVelocity<T, 3>>(native).get(i);
    gsl::at(magnetic_field, i) = get<MagneticField<T, 3>>(native).get(i);
    for (size_t j = 0; j < 3; ++j) {
      gsl::at(gsl::at(metric, i), j) =
          get<gr::Tags::SpatialMetric<T, 3>>(native).get(i, j);
    }
  }
  // Adapter consistency, not an independent physical reference for the kernel.
  const auto direct = hydro::primitive_sources::joint_sources(
      get(get<RestMassDensity<T>>(native)),
      get(get<SpecificInternalEnergy<T>>(native)),
      get(get<Pressure<T>>(native)), velocity, magnetic_field, metric);
  CHECK_ITERABLE_APPROX(get(get<Energy<T>>(all)), direct[0]);
  CHECK_ITERABLE_APPROX(get(get<Trace<T>>(all)), direct[4]);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(get<Momentum<T>>(all).get(i), gsl::at(direct, i + 1));
  }
}

template <typename Provider>
void check_scalar_and_vector(
    const Xcts::Solutions::WrappedGr<Provider, true>& wrapper,
    const std::array<std::array<double, 3>, 3>& points) {
  tnsr::I<DataVector, 3> batch{size_t{3}, 0.};
  for (size_t k = 0; k < 3; ++k) {
    tnsr::I<double, 3> scalar{};
    for (size_t i = 0; i < 3; ++i) {
      scalar.get(i) = gsl::at(gsl::at(points, k), i);
      batch.get(i)[k] = gsl::at(gsl::at(points, k), i);
    }
    check_sources(wrapper, scalar);
  }
  check_sources(wrapper, batch);
}

struct SentinelAccess : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct ProviderCalls {
  size_t enthalpy = 0;
  size_t lorentz = 0;
  bool fail = true;
  bool invalid_density = false;
};
class SentinelStar : public RelativisticEuler::Solutions::RotatingStar {
 public:
  using Base = RelativisticEuler::Solutions::RotatingStar;
  SentinelStar() = default;
  SentinelStar(std::string filename, std::shared_ptr<ProviderCalls> calls)
      : Base(std::move(filename), 100.), calls_(std::move(calls)) {}
  template <typename T, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<T, 3>& x, double time,
                                         tmpl::list<Tags...> /*meta*/) const {
    return {value<T, Tags>(x, time)...};
  }

 private:
  template <typename T, typename Tag>
  typename Tag::type value(const tnsr::I<T, 3>& x, double time) const {
    if constexpr (std::is_same_v<Tag, Enthalpy<T>>) {
      ++calls_->enthalpy;
      if (calls_->fail) {
        throw SentinelAccess("enthalpy requested");
      }
      return make_with_value<typename Tag::type>(x, 7.);
    } else if constexpr (std::is_same_v<Tag, Lorentz<T>>) {
      ++calls_->lorentz;
      if (calls_->fail) {
        throw SentinelAccess("Lorentz factor requested");
      }
      return make_with_value<typename Tag::type>(x, 1.25);
    } else {
      if constexpr (std::is_same_v<Tag, hydro::Tags::RestMassDensity<T>>) {
        if (calls_->invalid_density) {
          return make_with_value<typename Tag::type>(x, -1.);
        }
      }
      return get<Tag>(Base::variables(x, time, tmpl::list<Tag>{}));
    }
  }
  std::shared_ptr<ProviderCalls> calls_ = std::make_shared<ProviderCalls>();
};
}  // namespace

template <>
PUP::able::PUP_ID Xcts::Solutions::WrappedGr<SentinelStar, true>::my_PUP_ID = 0;

namespace {
template <typename T>
void test_forwarding(const T& exemplar) {
  auto calls = std::make_shared<ProviderCalls>();
  const Xcts::Solutions::WrappedGr<SentinelStar, true> wrapper{
      fixture("AnalyticSolutions/RelativisticEuler/RotatingStarId.dat"), calls};
  auto x = make_with_value<tnsr::I<T, 3>>(exemplar, 1.);
  CHECK_NOTHROW(
      wrapper.variables(x, tmpl::list<Energy<T>, Momentum<T>, Trace<T>>{}));
  CHECK(calls->enthalpy == 0);
  CHECK(calls->lorentz == 0);
  calls->fail = false;
  calls->invalid_density = true;
  const auto forwarded =
      wrapper.variables(x, tmpl::list<Enthalpy<T>, Lorentz<T>>{});
  CHECK(get(get<Enthalpy<T>>(forwarded)) == make_with_value<T>(exemplar, 7.));
  CHECK(get(get<Lorentz<T>>(forwarded)) == make_with_value<T>(exemplar, 1.25));
  CHECK(calls->enthalpy == 1);
  CHECK(calls->lorentz == 1);
  CHECK_THROWS_AS(wrapper.variables(x, tmpl::list<Energy<T>>{}),
                  std::domain_error);
  calls->invalid_density = false;
  CHECK_NOTHROW(wrapper.variables(
      x, tmpl::list<hydro::Tags::ComovingMagneticFieldSquared<T>>{}));
  CHECK(calls->lorentz == 2);
  CHECK(calls->enthalpy == 1);
  calls->fail = true;
  CHECK_THROWS_AS(wrapper.variables(x, tmpl::list<Enthalpy<T>>{}),
                  SentinelAccess);
  CHECK_THROWS_AS(
      wrapper.variables(
          x, tmpl::list<hydro::Tags::ComovingMagneticFieldSquared<T>>{}),
      SentinelAccess);
}

template <typename T>
[[noreturn]] Scalar<T> unexpected_provider_w() {
  throw SentinelAccess("unexpected provider W");
}

template <typename T>
void test_joint_cache(const T& exemplar) {
  using Computer = Xcts::Solutions::detail::WrappedGrVariables<T, true>;
  const auto x = make_with_value<tnsr::I<T, 3>>(exemplar, 0.);
  const auto geometry = gr::Solutions::Minkowski<3>{}.variables(
      x, 0., Xcts::Solutions::detail::gr_solution_vars<T, 3>{});
  tuples::tagged_tuple_from_typelist<
      Xcts::Solutions::detail::wrapped_gr_primitive_tags<T>>
      primitives;
  get<hydro::Tags::RestMassDensity<T>>(primitives) =
      Scalar<T>{make_with_value<T>(exemplar, 0.)};
  get<hydro::Tags::SpecificInternalEnergy<T>>(primitives) =
      Scalar<T>{make_with_value<T>(exemplar, -2.)};
  get<hydro::Tags::Pressure<T>>(primitives) =
      Scalar<T>{make_with_value<T>(exemplar, 0.)};
  for (size_t i = 0; i < 3; ++i) {
    get<hydro::Tags::SpatialVelocity<T, 3>>(primitives).get(i) =
        make_with_value<T>(exemplar, i == 0 ? 0.25 : 0.);
    get<hydro::Tags::MagneticField<T, 3>>(primitives).get(i) =
        make_with_value<T>(exemplar, i == 1 ? 2. : 0.);
  }
  const auto provider_w = &unexpected_provider_w<T>;
  typename Computer::Cache cache{get_size(exemplar)};
  const Computer computer{std::nullopt, std::nullopt, x,
                          geometry,     primitives,   provider_w};
  const auto energy = cache.get_var(computer, Energy<T>{});
  CHECK(get(energy) == make_with_value<T>(exemplar, 2.125));
  // Each tag has its own CachedTempBuffer flag. Mutating rho after E must
  // not cause S or trace to recompute the joint state from invalid primitives.
  get<hydro::Tags::RestMassDensity<T>>(primitives) =
      Scalar<T>{make_with_value<T>(exemplar, -1.)};
  const auto momentum = cache.get_var(computer, Momentum<T>{});
  CHECK(get<0>(momentum) == make_with_value<T>(exemplar, 1.));
  CHECK(get<1>(momentum) == make_with_value<T>(exemplar, 0.));
  CHECK(get<2>(momentum) == make_with_value<T>(exemplar, 0.));
  CHECK(cache.get_var(computer, Trace<T>{}) == energy);
  typename Computer::Cache fresh_cache{get_size(exemplar)};
  const Computer fresh{std::nullopt, std::nullopt, x,
                       geometry,     primitives,   provider_w};
  CHECK_THROWS_AS(fresh_cache.get_var(fresh, Energy<T>{}), std::domain_error);
}
}  // namespace

// [[Timeout, 20]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.WrappedGrSources",
    "[Unit][PointwiseFunctions]") {
  using Xcts::Solutions::WrappedGr;
  const WrappedGr<grmhd::AnalyticData::CcsnCollapse, true> ccsn{
      fixture("AnalyticData/GrMhd/CcsnCollapseID.dat"),   1.e-5, 339., 100.,
      fixture("Hydro/EquationsOfState/dd2_unit_test.h5"), "dd2"};
  check_scalar_and_vector(
      ccsn, {{{0.01, 0.01, 0.01}, {0.02, 0.01, 0.005}, {0.003, 0.007, 0.01}}});
  const WrappedGr<RelativisticEuler::Solutions::RotatingStar, true> rotating{
      fixture("AnalyticSolutions/RelativisticEuler/RotatingStarId.dat"), 100.};
  const double radius = rotating.gr_solution().equatorial_radius();
  check_scalar_and_vector(rotating,
                          {{{0.2 * radius, 0.03 * radius, 0.02 * radius},
                            {radius, 0.1 * radius, 0.1 * radius},
                            {1.5 * radius, 0.1 * radius, 0.2 * radius}}});
  std::vector<std::unique_ptr<
      grmhd::AnalyticData::InitialMagneticFields::InitialMagneticField>>
      fields;
  fields.push_back(
      std::make_unique<grmhd::AnalyticData::InitialMagneticFields::Poloidal>(
          size_t{2}, 6.5536e-6, 2500., std::array<double, 3>{{0., 0., 0.}},
          100.));
  const WrappedGr<grmhd::AnalyticData::MagnetizedTovStar, true> tov{
      1.28e-3,
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(100., 2.),
      RelativisticEuler::Solutions::TovCoordinates::Schwarzschild,
      std::move(fields)};
  check_scalar_and_vector(tov, {{{1., 0.2, 0.3}, {4., 1., 2.}, {20., 2., 3.}}});
}

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.WrappedGrForwarding",
    "[Unit][PointwiseFunctions]") {
  test_forwarding(0.);
  test_forwarding(DataVector(3, 0.));
}

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.WrappedGrCache",
    "[Unit][PointwiseFunctions]") {
  test_joint_cache(0.);
  test_joint_cache(DataVector(3, 0.));
}
