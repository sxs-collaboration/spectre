// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"

#include <array>
#include <memory>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::creators::time_dependent_options {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
template <bool AllowSettleFoTs>
ExpansionMapOptions<AllowSettleFoTs>::ExpansionMapOptions(
    const std::array<double, 3>& initial_values_in,
    double decay_timescale_outer_boundary_in,
    const std::array<double, 3>& initial_values_outer_boundary_in,
    double decay_timescale_in, const Options::Context& context)
    : decay_timescale_outer_boundary(decay_timescale_outer_boundary_in),
      decay_timescale(decay_timescale_in) {
  if constexpr (AllowSettleFoTs) {
    initial_values = std::array{DataVector{initial_values_in[0]},
                                DataVector{initial_values_in[1]},
                                DataVector{initial_values_in[2]}};
    initial_values_outer_boundary =
        std::array{DataVector{initial_values_outer_boundary_in[0]},
                   DataVector{initial_values_outer_boundary_in[1]},
                   DataVector{initial_values_outer_boundary_in[2]}};
  } else {
    PARSE_ERROR(context,
                "This class does not allow SettleToConst functions of time, "
                "but the constructor for allowing SettleToConst functions of "
                "time was used. Please use the other constructor.");
  }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)

template <bool AllowSettleFoTs>
ExpansionMapOptions<AllowSettleFoTs>::ExpansionMapOptions(
    const std::array<double, 3>& initial_values_in,
    double decay_timescale_outer_boundary_in,
    double asymptotic_velocity_outer_boundary_in,
    const Options::Context& /*context*/)
    : decay_timescale_outer_boundary(decay_timescale_outer_boundary_in),
      asymptotic_velocity_outer_boundary(
          asymptotic_velocity_outer_boundary_in) {
  initial_values = std::array{DataVector{initial_values_in[0]},
                              DataVector{initial_values_in[1]},
                              DataVector{initial_values_in[2]}};
  initial_values_outer_boundary =
      std::array{DataVector{1.0}, DataVector{0.0}, DataVector{0.0}};
}

template <bool AllowSettleFoTs>
FunctionsOfTimeMap get_expansion(
    const std::variant<ExpansionMapOptions<AllowSettleFoTs>, FromVolumeFile>&
        expansion_map_options,
    const double initial_time, const double expiration_time) {
  const std::string name{"Expansion"};
  const std::string name_outer_boundary{"ExpansionOuterBoundary"};
  FunctionsOfTimeMap result{};

  if (std::holds_alternative<FromVolumeFile>(expansion_map_options)) {
    const auto& from_vol_file = std::get<FromVolumeFile>(expansion_map_options);
    auto volume_fot = from_vol_file.retrieve_function_of_time(
        {name, name_outer_boundary}, initial_time);

    // Expansion must be either a PiecewisePolynomial or a SettleToConstant
    const auto* pp_volume_fot =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
            volume_fot.at(name).get());
    const auto* settle_volume_fot =
        dynamic_cast<domain::FunctionsOfTime::SettleToConstant*>(
            volume_fot.at(name).get());

    if (UNLIKELY(pp_volume_fot == nullptr and settle_volume_fot == nullptr)) {
      ERROR_NO_TRACE(
          "Expansion function of time read from volume data is not a "
          "PiecewisePolynomial<2> or a SettleToConstant. Cannot use it to "
          "initialize the expansion map.");
    }

    if (from_vol_file.replay()) {
      result[name] = std::move(volume_fot.at(name));
    } else {
      result[name] =
          volume_fot.at(name)->create_at_time(initial_time, expiration_time);
    }

    // Outer boundary must be either a FixedSpeedCubic or a SettleToConstant
    const auto* outer_boundary_cubic_volume_fot =
        dynamic_cast<domain::FunctionsOfTime::FixedSpeedCubic*>(
            volume_fot.at(name_outer_boundary).get());
    const auto* outer_boundary_settle_volume_fot =
        dynamic_cast<domain::FunctionsOfTime::SettleToConstant*>(
            volume_fot.at(name_outer_boundary).get());

    if (UNLIKELY(outer_boundary_cubic_volume_fot == nullptr and
                 outer_boundary_settle_volume_fot == nullptr)) {
      ERROR_NO_TRACE(
          "ExpansionOuterBoundary function of time read from volume data is "
          "not a FixedSpeedCubic or a SettleToConstant. Cannot use it to "
          "initialize the expansion map.");
    }

    ASSERT(outer_boundary_cubic_volume_fot != nullptr or AllowSettleFoTs,
           "ExpansionOuterBoundary function of time in the volume file is a "
           "SettleToConstant, but SettleToConstant functions of time aren't "
           "allowed.");

    if (from_vol_file.replay()) {
      result[name] = std::move(volume_fot.at(name_outer_boundary));
    } else {
      result[name_outer_boundary] =
          volume_fot.at(name_outer_boundary)->get_clone();
    }
  } else if (std::holds_alternative<ExpansionMapOptions<AllowSettleFoTs>>(
                 expansion_map_options)) {
    const auto& hard_coded_options =
        std::get<ExpansionMapOptions<AllowSettleFoTs>>(expansion_map_options);

    if (hard_coded_options.asymptotic_velocity_outer_boundary.has_value()) {
      result[name] =
          std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
              initial_time, hard_coded_options.initial_values, expiration_time);
      result[name_outer_boundary] =
          std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
              hard_coded_options.initial_values_outer_boundary[0][0],
              initial_time,
              hard_coded_options.asymptotic_velocity_outer_boundary.value(),
              hard_coded_options.decay_timescale_outer_boundary);
    } else {
      ASSERT(hard_coded_options.decay_timescale.has_value(),
             "To construct an ExpansionMap SettleToConstant function of time, "
             "a decay timescale must be supplied.");
      result[name] =
          std::make_unique<domain::FunctionsOfTime::SettleToConstant>(
              hard_coded_options.initial_values, initial_time,
              hard_coded_options.decay_timescale.value());
      result[name_outer_boundary] =
          std::make_unique<domain::FunctionsOfTime::SettleToConstant>(
              hard_coded_options.initial_values_outer_boundary, initial_time,
              hard_coded_options.decay_timescale_outer_boundary);
    }
  } else {
    ERROR("Unknown ExpansionMap.");
  }

  return result;
}

#define ALLOWSETTLE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                     \
  template struct ExpansionMapOptions<ALLOWSETTLE(data)>;        \
  template FunctionsOfTimeMap get_expansion(                     \
      const std::variant<ExpansionMapOptions<ALLOWSETTLE(data)>, \
                         FromVolumeFile>& expansion_map_options, \
      double initial_time, double expiration_time);

GENERATE_INSTANTIATIONS(INSTANTIATE, (true, false))

#undef INSTANTIATE
#undef ALLOWSETTLE
}  // namespace domain::creators::time_dependent_options
