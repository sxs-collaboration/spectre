// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"

#include <array>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::creators::time_dependent_options {
template <size_t Dim>
TranslationMapOptions<Dim>::TranslationMapOptions(
    std::variant<std::array<std::array<double, Dim>, 3>,
                 FromVolumeFile<names::Translation>>
        values_from_options) {
  if (std::holds_alternative<std::array<std::array<double, Dim>, 3>>(
          values_from_options)) {
    auto& values =
        std::get<std::array<std::array<double, Dim>, 3>>(values_from_options);
    for (size_t i = 0; i < initial_values.size(); i++) {
      gsl::at(initial_values, i) = DataVector{Dim, 0.0};
      for (size_t j = 0; j < Dim; j++) {
        gsl::at(initial_values, i)[j] = gsl::at(gsl::at(values, i), j);
      }
    }
  } else if (std::holds_alternative<FromVolumeFile<names::Translation>>(
                 values_from_options)) {
    auto& values_from_file =
        std::get<FromVolumeFile<names::Translation>>(values_from_options);
    initial_values = values_from_file.values;
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class TranslationMapOptions<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace domain::creators::time_dependent_options
