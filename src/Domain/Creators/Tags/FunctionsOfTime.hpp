// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/OptionTags.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/ReadSpecPiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

namespace detail {

CREATE_HAS_STATIC_MEMBER_VARIABLE(override_functions_of_time)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(override_functions_of_time)

template <typename Metavariables, bool HasOverrideCubicFunctionsOfTime>
struct OptionList {
  using type = tmpl::conditional_t<
      Metavariables::override_functions_of_time,
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
/// (if `override_functions_of_time` is true in the metavariables) read
/// from a file.
///
/// \details When `override_functions_of_time == true` in the
/// metavariables, after obtaining the FunctionsOfTime from the DomainCreator,
/// one or more of those FunctionsOfTime (which must be cubic piecewise
/// polynomials) is overriden using data read from an HDF5 file via
/// domain::Tags::read_spec_piecewise_polynomial()
struct FunctionsOfTimeInitialize : FunctionsOfTime, db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  static constexpr bool pass_metavariables = true;

  static std::string name() { return "FunctionsOfTime"; }

  template <typename Metavariables>
  using option_tags = typename ::detail::OptionList<
      Metavariables,
      ::detail::has_override_functions_of_time_v<Metavariables>>::type;

  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const std::optional<std::string>& function_of_time_file,
      const std::map<std::string, std::string>& function_of_time_name_map) {
    auto functions_of_time = domain_creator->functions_of_time();

    if (function_of_time_file.has_value()) {
      domain::FunctionsOfTime::override_functions_of_time(
          make_not_null(&functions_of_time), *function_of_time_file,
          function_of_time_name_map);
    }

    return functions_of_time;
  }

  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) {
    return domain_creator->functions_of_time();
  }
};
}  // namespace domain::Tags
