// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::dg {
/// Option tags related to elliptic discontinuous Galerkin schemes
namespace OptionTags {

struct Discretization {
  static constexpr Options::String help =
      "Options for the discretization of the elliptic equations";
};

struct DiscontinuousGalerkin {
  using group = Discretization;
  static constexpr Options::String help =
      "Options for the discontinuous Galerkin discretization";
};

struct PenaltyParameter {
  using type = double;
  static constexpr Options::String help =
      "The prefactor to the penalty term of the numerical flux. Values closer "
      "to one lead to better-conditioned problems, but on curved meshes the "
      "penalty parameter may need to be increased to keep the problem "
      "well-defined.";
  using group = DiscontinuousGalerkin;
};

struct Massive {
  using type = bool;
  static constexpr Options::String help =
      "Whether or not to multiply the DG operator with the mass matrix. "
      "Massive DG operators can be easier to solve because they are symmetric, "
      "or at least closer to symmetry";
  using group = DiscontinuousGalerkin;
};

}  // namespace OptionTags

/// DataBox tags related to elliptic discontinuous Galerkin schemes
namespace Tags {

/*!
 * \brief The prefactor to the penalty term of the numerical flux
 *
 * \see elliptic::dg::penalty
 */
struct PenaltyParameter : db::SimpleTag {
  using type = double;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::PenaltyParameter>;
  static double create_from_options(const double value) { return value; }
};

/// Whether or not to multiply the DG operator with the mass matrix. Massive DG
/// operators can be easier to solve because they are symmetric, or at least
/// closer to symmetry.
struct Massive : db::SimpleTag {
  using type = bool;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::Massive>;
  static bool create_from_options(const bool value) { return value; }
};

}  // namespace Tags
}  // namespace elliptic::dg
