// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DiscontinuousGalerkin/OptionTags.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
/*!
 * \brief The coefficients \f$(a, b, c)\f$ that give the default number of
 * finite-difference grid points \f$2N-1\f$ per dimension.
 *
 * See `number_of_subcell_points()` for how the coefficients are used.
 */
constexpr std::array<int, 3> auto_number_of_subcell_points_coefficients{
    {0, 2, -1}};

/*!
 * \brief Computes the number of cell-centered finite-difference grid points
 * \f$n_\mathrm{FD} = a N^2 + b N + c\f$ in a dimension with \f$N\f$ DG grid
 * points, where \f$(a, b, c)\f$ are the `number_of_points_coefficients`.
 *
 * It is an error if \f$n_\mathrm{FD} < N\f$, since then the FD solution
 * cannot be reconstructed on the DG grid, or if \f$n_\mathrm{FD}\f$ is outside
 * the supported range of finite-difference grid points.
 *
 * \note This lives outside the `DgSubcell` library so that it can also be used
 * by code that must not depend on `DgSubcell`, such as the element
 * distribution (see `evolution::dg::compute_weighting_extents_override`).
 */
size_t number_of_subcell_points(
    size_t num_dg_points,
    const std::array<int, 3>& number_of_points_coefficients);

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \brief The coefficients \f$(a, b, c)\f$ passed to
/// `evolution::dg::number_of_subcell_points()` and
/// `evolution::dg::subcell::fd::mesh()`.
///
/// See `evolution::dg::OptionTags::NumberOfSubcellPoints`.
struct NumberOfSubcellPointsCoefficients : db::SimpleTag {
  using type = std::array<int, 3>;
  using option_tags = tmpl::list<OptionTags::NumberOfSubcellPoints>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::optional<std::array<int, 3>>& number_of_points_coefficients) {
    return number_of_points_coefficients.value_or(
        auto_number_of_subcell_points_coefficients);
  }
};
}  // namespace Tags
}  // namespace evolution::dg
