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
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/ReadSpecPiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Options/Options.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

/// \cond
template <size_t VolumeDim>
class DomainCreator;
/// \endcond

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
/// Tag to retreive the FunctionsOfTime from the GlobalCache.
struct FunctionsOfTime : db::BaseTag {};

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
    if (function_of_time_file) {
      // Currently, only support order 2 or 3 piecewise polynomials.
      // This could be generalized later, but the SpEC functions of time
      // that we will read in with this action will always be 2nd-order or
      // 3rd-order piecewise polynomials
      std::unordered_map<std::string,
                         domain::FunctionsOfTime::PiecewisePolynomial<2>>
          spec_functions_of_time_second_order{};
      std::unordered_map<std::string,
                         domain::FunctionsOfTime::PiecewisePolynomial<3>>
          spec_functions_of_time_third_order{};
      std::unordered_map<std::string,
                         domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>
          spec_functions_of_time_quaternion{};

      // Import those functions of time of each supported order
      domain::FunctionsOfTime::read_spec_piecewise_polynomial(
          make_not_null(&spec_functions_of_time_second_order),
          *function_of_time_file, function_of_time_name_map);
      domain::FunctionsOfTime::read_spec_piecewise_polynomial(
          make_not_null(&spec_functions_of_time_third_order),
          *function_of_time_file, function_of_time_name_map);

      auto functions_of_time{domain_creator->functions_of_time()};

      bool uses_quaternion_rotation = false;
      for (const auto& name_and_fot : functions_of_time) {
        auto* maybe_quaternion_fot =
            dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
                name_and_fot.second.get());
        if (maybe_quaternion_fot != nullptr) {
          uses_quaternion_rotation = true;
        }
      }

      // Only parse as quaternion function of time if it exists
      if (uses_quaternion_rotation) {
        domain::FunctionsOfTime::read_spec_piecewise_polynomial(
            make_not_null(&spec_functions_of_time_quaternion),
            *function_of_time_file, function_of_time_name_map, true);
      }

      for (const auto& [spec_name, spectre_name] : function_of_time_name_map) {
        (void)spec_name;
        // The FunctionsOfTime we are mutating must already have
        // an element with key==spectre_name; this action only
        // mutates the value associated with that key
        if (functions_of_time.count(spectre_name) == 0) {
          ERROR("Trying to import data for key "
                << spectre_name
                << " in FunctionsOfTime, but FunctionsOfTime does not "
                   "contain that key. This might happen if the option "
                   "FunctionOfTimeNameMap is not specified correctly. Keys "
                   "contained in FunctionsOfTime: "
                << keys_of(functions_of_time) << "\n");
        }
        auto* piecewise_polynomial_second_order =
            dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
                functions_of_time[spectre_name].get());
        auto* piecewise_polynomial_third_order =
            dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<3>*>(
                functions_of_time[spectre_name].get());
        auto* quaternion_fot_third_order =
            dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
                functions_of_time[spectre_name].get());
        if (piecewise_polynomial_second_order == nullptr) {
          if (piecewise_polynomial_third_order == nullptr) {
            if (quaternion_fot_third_order == nullptr) {
              ERROR("The function of time with name "
                    << spectre_name
                    << " is not a PiecewisePolynomial<2>, "
                       "PiecewisePolynomial<3>, or QuaternionFunctionOfTime<3> "
                       "and so cannot be set using "
                       "read_spec_piecewise_polynomial\n");
            } else {
              *quaternion_fot_third_order =
                  spec_functions_of_time_quaternion.at(spectre_name);
            }
          } else {
            *piecewise_polynomial_third_order =
                spec_functions_of_time_third_order.at(spectre_name);
          }
        } else {
          *piecewise_polynomial_second_order =
              spec_functions_of_time_second_order.at(spectre_name);
        }
      }

      return functions_of_time;
    } else {
      return domain_creator->functions_of_time();
    }
  }

  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) {
    return domain_creator->functions_of_time();
  }
};
}  // namespace domain::Tags
