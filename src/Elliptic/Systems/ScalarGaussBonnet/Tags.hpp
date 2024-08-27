// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the ScalarGaussBonnet system.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
class DataVector;
/// \endcond

namespace sgb {
/// Tags related to solving the scalar equation in sGB gravity.
namespace OptionTags {

struct OptionGroup {
  static std::string name() { return "ScalarGaussBonnet"; }
  static constexpr Options::String help =
      "Options for elliptic solve in sGB gravity.";
};

struct Epsilon2 {
  static std::string name() { return "Epsilon2"; }
  using type = double;
  using group = OptionGroup;
  static constexpr Options::String help{
      "Epsilon2 (quadratic term) for the coupling function."};
};

struct Epsilon4 {
  static std::string name() { return "Epsilon4"; }
  using type = double;
  using group = OptionGroup;
  static constexpr Options::String help{
      "Epsilon4 (quartic term) for the coupling function."};
};

struct RolloffLocation {
  static std::string name() { return "RolloffLocation"; }
  using type = double;
  using group = OptionGroup;
  static constexpr Options::String help{
      "Location of center of tanh function for rolling off the shift."};
};

struct RolloffRate {
  static std::string name() { return "RolloffRate"; }
  using type = double;
  using group = OptionGroup;
  static constexpr Options::String help{
      "Rate of roll off for rolling off the shift."};
};

}  // namespace OptionTags
namespace Tags {

/*!
 * \brief Epsilon2 (quadratic term) for the coupling function.
 */
struct Epsilon2 : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Epsilon2>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double epsilon) { return epsilon; }
};

/*!
 * \brief Epsilon4 (quartic term) for the coupling function.
 */
struct Epsilon4 : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Epsilon4>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double epsilon) { return epsilon; }
};

/*!
 * \brief Location of center of tanh function for rolling off the shift.
 */
struct RolloffLocation : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::RolloffLocation>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double rolloff_location) {
    return rolloff_location;
  }
};

/*!
 * \brief Rate of roll off for rolling off the shift.
 */
struct RolloffRate : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::RolloffRate>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double rolloff_rate) {
    return rolloff_rate;
  }
};

/*!
 * \brief The scalar field $\Psi$.
 */
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Rolled-off shift (i.e. the shift used in computing the fluxes).
 */
struct RolledOffShift : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};

/*!
 * \brief Pi computed using the rolled-off shift.
 */
struct PiWithRolledOffShift : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace Tags
}  // namespace sgb
