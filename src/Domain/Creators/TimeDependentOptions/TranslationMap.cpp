// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"

#include <array>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::creators::time_dependent_options {
template <size_t Dim>
TranslationMapOptions<Dim>::TranslationMapOptions(
    const std::array<std::array<double, Dim>, 3>& initial_values_in,
    const Options::Context& context) {
  if (initial_values_in.empty() or initial_values_in.size() > 3) {
    PARSE_ERROR(
        context,
        "Must specify at least the value of the translation, and optionally "
        "up to 2 time derivatives.");
  }

  for (size_t i = 0; i < initial_values_in.size(); i++) {
    gsl::at(initial_values, i) = DataVector{gsl::at(initial_values_in, i)};
  }
}

template <size_t Dim>
std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> get_translation(
    const std::variant<TranslationMapOptions<Dim>, FromVolumeFile>&
        translation_map_options,
    const double initial_time, const double expiration_time) {
  const std::string name = "Translation";
  std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> result{};

  if (std::holds_alternative<FromVolumeFile>(translation_map_options)) {
    const auto& from_vol_file =
        std::get<FromVolumeFile>(translation_map_options);
    auto volume_fot =
        from_vol_file.retrieve_function_of_time({name}, initial_time);

    // It must be a PiecewisePolynomial
    if (UNLIKELY(dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
                     volume_fot.at(name).get()) == nullptr)) {
      ERROR_NO_TRACE(
          "Translation function of time read from volume data is not a "
          "PiecewisePolynomial<2>. Cannot use it to initialize the translation "
          "map.");
    }

    if (from_vol_file.replay()) {
      result = std::move(volume_fot.at(name));
    } else {
      result =
          volume_fot.at(name)->create_at_time(initial_time, expiration_time);
    }
  } else if (std::holds_alternative<TranslationMapOptions<Dim>>(
                 translation_map_options)) {
    const auto& hard_coded_options =
        std::get<TranslationMapOptions<Dim>>(translation_map_options);

    result = std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
        initial_time, hard_coded_options.initial_values, expiration_time);
  } else {
    ERROR("Unknown TranslationMap.");
  }

  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template class TranslationMapOptions<DIM(data)>;                             \
  template std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>            \
  get_translation(const std::variant<TranslationMapOptions<DIM(data)>,         \
                                     FromVolumeFile>& translation_map_options, \
                  double initial_time, double expiration_time);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace domain::creators::time_dependent_options
