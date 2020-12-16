// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/OptionTags.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/ReadSpecThirdOrderPiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

/// \cond
template <size_t VolumeDim>
class DomainCreator;
/// \endcond

namespace detail {

CREATE_HAS_STATIC_MEMBER_VARIABLE(override_cubic_functions_of_time)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(override_cubic_functions_of_time)

template <typename Metavariables, bool HasOverrideCubicFunctionsOfTime>
struct OptionList {
  using type = tmpl::conditional_t<
      Metavariables::override_cubic_functions_of_time,
      tmpl::list<domain::OptionTags::DomainCreator<Metavariables::volume_dim>,
                 domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
                 domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>,
      tmpl::list<domain::OptionTags::DomainCreator<Metavariables::volume_dim>>>;
};

template <typename Metavariables>
struct OptionList<Metavariables, false> {
  using type =
      tmpl::list<domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;
};
}  // namespace detail

namespace domain::Tags {
/// \brief The FunctionsOfTime initialized from a DomainCreator or
/// (if `override_cubic_functions_of_time` is true in the metavariables) read
/// from a file.
///
/// \details When `override_cubic_functions_of_time == true` in the
/// metavariables, after obtaining the FunctionsOfTime from the DomainCreator,
/// one or more of those FunctionsOfTime (which must be cubic piecewise
/// polynomials) is overriden using data read from an HDF5 file via
/// domain::Tags::read_spec_third_order_piecewise_polynomial()
struct FunctionsOfTime : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  using option_tags = typename ::detail::OptionList<
      Metavariables,
      ::detail::has_override_cubic_functions_of_time_v<Metavariables>>::type;

  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const std::optional<std::string>& function_of_time_file,
      const std::optional<std::map<std::string, std::string>>&
          function_of_time_name_map) noexcept {
    if (function_of_time_file and function_of_time_name_map) {
      // Currently, only support order 3 piecewise polynomials.
      // This could be generalized later, but the SpEC functions of time
      // that we will read in with this action will always be 3rd-order
      // piecewise polynomials
      constexpr size_t max_deriv{3};
      std::unordered_map<
          std::string, domain::FunctionsOfTime::PiecewisePolynomial<max_deriv>>
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

  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) noexcept {
    return domain_creator->functions_of_time();
  }
};
}  // namespace domain::Tags
