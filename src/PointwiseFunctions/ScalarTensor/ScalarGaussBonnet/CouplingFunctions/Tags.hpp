// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/CouplingFunction.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace ScalarTensor::sgb::OptionTags {
struct Group;
}  // namespace ScalarTensor::sgb::OptionTags
/// \endcond

namespace ScalarTensor::sgb {

namespace OptionTags {
/*!
 * \brief Group encoding all information about the coupling term in the action
 * of Einstein-scalar-Gauss-Bonnet gravity.
 *
 * \details The coupling term considered here is consistent with the expression
 * of the action in \cite Nee2024bur , and reads
 * \f$\ell^2 F[\Psi] \mathcal{G}\f$, where \f$\ell\f$ is a coupling constant
 * encoding the length below which modifications to General Relativity start
 * becoming relevant, \f$F[\Psi]\f$ is the coupling function and
 * \f$\mathcal{G}\f$ is the Gauss-Bonnet invariant.
 */
struct CouplingTerm {
  static constexpr Options::String help = {
      "Group containing information about the coupling term in the action"};
};

/*!
 * \brief Coupling constant appearing in the coupling term of
 * Einstein-scalar-Gauss-Bonnet gravity.
 *
 * \details This coupling constant has the dimensions of a length and
 * encodes the length scale below which modifications to General Relativity
 * start becoming relevant.
 */
struct Ell {
  using type = double;
  static constexpr Options::String help = {
      "Coupling constant setting the length scales where modifications enter"};
  using group = CouplingTerm;
};

/*!
 * \brief Coupling function \f$F[\Psi]\f$.
 */
struct CouplingFunction {
  using type =
      std::unique_ptr<ScalarTensor::sgb::CouplingFunctions::CouplingFunction>;
  static constexpr Options::String help = {
      "The coupling function in the scalar-Gauss-Bonnet term"};
  using group = CouplingTerm;
};

}  // namespace OptionTags

namespace Tags {
/*!
 * \brief Coupling constant appearing in the coupling term of
 * Einstein-scalar-Gauss-Bonnet gravity.
 *
 * \details This coupling constant has the dimensions of a length and
 * encodes the length scale below which modifications to General Relativity
 * start becoming relevant.
 */
struct Ell : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Ell>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double value) { return value; }
};

/*!
 * \brief Coupling function \f$F[\Psi]\f$.
 */
struct CouplingFunction : db::SimpleTag {
  using type =
      std::unique_ptr<ScalarTensor::sgb::CouplingFunctions::CouplingFunction>;
  using option_tags = tmpl::list<OptionTags::CouplingFunction>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return serialize_and_deserialize<type>(value);
  }
};

}  // namespace Tags

}  // namespace ScalarTensor::sgb
