// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/PrettyType.hpp"

/// Functionality related to solving elliptic partial differential equations
namespace elliptic {
namespace Tags {

/*!
 * \brief Holds an object that computes the principal part of the elliptic PDEs.
 *
 * \details The `FluxesComputerType` must have an `apply` function that computes
 * fluxes from the system fields.
 *
 * When placed in the cache, the `FluxesComputerType` is default-constructed.
 * Provide tags that derive from this tag to construct it differently, e.g. to
 * construct it from problem-specific options or retrieve it from an analytic
 * solution.
 */
template <typename FluxesComputerType>
struct FluxesComputer : db::SimpleTag {
  using type = FluxesComputerType;
  static std::string name() noexcept {
    return pretty_type::short_name<FluxesComputerType>();
  }
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static FluxesComputerType create_from_options() {
    return FluxesComputerType{};
  }
};

}  // namespace Tags
}  // namespace elliptic
