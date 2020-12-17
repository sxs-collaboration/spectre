// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/Tags.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/ReadSpecThirdOrderPiecewisePolynomial.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::Tags {
template <size_t Dim>
auto InitialFunctionsOfTime<Dim, true>::create_from_options(
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
    const std::optional<std::string>& function_of_time_file,
    const std::optional<std::map<std::string, std::string>>&
        function_of_time_name_map) noexcept -> type {
  if (function_of_time_file and function_of_time_name_map) {
    // Currently, only support order 3 piecewise polynomials.
    // This could be generalized later, but the SpEC functions of time
    // that we will read in with this action will always be 3rd-order
    // piecewise polynomials
    constexpr size_t max_deriv{3};
    std::unordered_map<std::string,
                       domain::FunctionsOfTime::PiecewisePolynomial<max_deriv>>
        spec_functions_of_time{};
    domain::FunctionsOfTime::read_spec_third_order_piecewise_polynomial(
        make_not_null(&spec_functions_of_time), *function_of_time_file,
        *function_of_time_name_map);

    auto functions_of_time{domain_creator->functions_of_time()};
    for (const auto& [spec_name, spectre_name] : *function_of_time_name_map) {
      (void)spec_name;
      // The FunctionsOfTime we are mutating must already have
      // an element with key==spectre_name; this action only
      // mutates the value associated with that key
      if (functions_of_time.count(spectre_name) == 0) {
        std::vector<std::string> keys_in_functions_of_time{
            keys_of(functions_of_time)};
        ERROR("Trying to import data for key "
              << spectre_name
              << "in FunctionsOfTime, but FunctionsOfTime does not "
                 "contain that key. This might happen if the option "
                 "FunctionOfTimeNameMap is not specified correctly. Keys "
                 "contained in FunctionsOfTime: "
              << keys_in_functions_of_time << "\n");
      }
      auto* piecewise_polynomial = dynamic_cast<
          domain::FunctionsOfTime::PiecewisePolynomial<max_deriv>*>(
          functions_of_time[spectre_name].get());
      if (piecewise_polynomial == nullptr) {
        ERROR("The function of time with name "
              << spectre_name << " is not a PiecewisePolynomial<" << max_deriv
              << "> and so cannot be set using "
                 "ReadSpecThirdOrderPiecewisePolynomial\n");
      }
      *piecewise_polynomial = spec_functions_of_time.at(spectre_name);
    }

    return functions_of_time;
  } else {
    return domain_creator->functions_of_time();
  }
}

template <size_t Dim>
auto InitialFunctionsOfTime<Dim, false>::create_from_options(
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) noexcept
    -> type {
  return domain_creator->functions_of_time();
}
}  // namespace domain::Tags

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                   \
  template auto                                                                \
  domain::Tags::InitialFunctionsOfTime<DIM(data), true>::create_from_options(  \
      const std::unique_ptr<DomainCreator<DIM(data)>>& domain_creator,         \
      const std::optional<std::string>& function_of_time_file,                 \
      const std::optional<std::map<std::string, std::string>>&                 \
          function_of_time_name_map) noexcept->type;                           \
  template auto                                                                \
  domain::Tags::InitialFunctionsOfTime<DIM(data), false>::create_from_options( \
      const std::unique_ptr<DomainCreator<DIM(data)>>&                         \
          domain_creator) noexcept->type;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
