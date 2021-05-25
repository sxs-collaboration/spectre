// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace NewtonianEuler::subcell {
/*!
 * \brief The troubled-cell indicator run on DG initial data to see if we need
 * to switch to subcell.
 *
 * Uses the two-mesh relaxed discrete maximum principle as well as the Persson
 * TCI applied to the mass density and pressure.
 */
template <size_t Dim>
struct DgInitialDataTci {
 private:
  using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
  using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;
  using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<Dim>;
  using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;
  template <typename Tag>
  using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;

 public:
  using argument_tags = tmpl::list<domain::Tags::Mesh<Dim>, Pressure>;

  static bool apply(
      const Variables<
          tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>>& dg_vars,
      const Variables<
          tmpl::list<Inactive<MassDensityCons>, Inactive<MomentumDensity>,
                     Inactive<EnergyDensity>>>& subcell_vars,
      double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
      const Mesh<Dim>& dg_mesh, const Scalar<DataVector>& dg_pressure) noexcept;
};
}  // namespace NewtonianEuler::subcell
