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
#include "Domain/OptionTags.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class DomainCreator;
/// \endcond

namespace domain::Tags {
/// \brief The FunctionsOfTime obtained from a DomainCreator or
/// (if `OverrideCubicFunctionsOfTime` is true) read from a file.
///
/// \details When `OverrideCubicFunctionsOfTime` is true, after obtaining the
/// FunctionsOfTime from the DomainCreator, one or more of those
/// FunctionsOfTime (which must be cubic piecewise polynomials) is overriden
/// using data read from an HDF5 file via
/// domain::Tags::read_spec_third_order_piecewise_polynomial()
template <size_t Dim, bool OverrideCubicFunctionsOfTime = false>
struct InitialFunctionsOfTime {};

/// The FunctionsOfTime obtained from a DomainCreator, with
/// cubic piecewise polynomial functions overriden using
/// domain::Tags::read_spec_third_order_piecewise_polynomial()
template <size_t Dim>
struct InitialFunctionsOfTime<Dim, true> : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
  using option_tags =
      tmpl::list<domain::OptionTags::DomainCreator<Dim>,
                 domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
                 domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
      const std::optional<std::string>& function_of_time_file,
      const std::optional<std::map<std::string, std::string>>&
          function_of_time_name_map) noexcept;
};

/// The functions of time obtained from a domain creator
template <size_t Dim>
struct InitialFunctionsOfTime<Dim, false> : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) noexcept;
};

/// The functions of time
struct FunctionsOfTime : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
};
}  // namespace domain::Tags
