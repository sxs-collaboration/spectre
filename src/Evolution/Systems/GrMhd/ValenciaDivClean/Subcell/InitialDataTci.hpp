// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief The troubled-cell indicator run on DG initial data to see if we need
 * to switch to subcell.
 *
 * Uses the two-mesh relaxed discrete maximum principle as well as the Persson
 * TCI applied to \f$\tilde{D}\f$ and \f$\tilde{\tau}\f$.
 */
struct DgInitialDataTci {
 private:
  template <typename Tag>
  using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;

 public:
  using argument_tags = tmpl::list<domain::Tags::Mesh<3>>;

  static bool apply(
      const Variables<tmpl::list<
          ValenciaDivClean::Tags::TildeD, ValenciaDivClean::Tags::TildeTau,
          ValenciaDivClean::Tags::TildeS<>, Tags::TildeB<>,
          ValenciaDivClean::Tags::TildePhi>>& dg_vars,
      const Variables<tmpl::list<Inactive<ValenciaDivClean::Tags::TildeD>,
                                 Inactive<ValenciaDivClean::Tags::TildeTau>,
                                 Inactive<ValenciaDivClean::Tags::TildeS<>>,
                                 Inactive<ValenciaDivClean::Tags::TildeB<>>,
                                 Inactive<ValenciaDivClean::Tags::TildePhi>>>&
          subcell_vars,
      double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
      const Mesh<3>& dg_mesh) noexcept;
};
}  // namespace grmhd::ValenciaDivClean::subcell
