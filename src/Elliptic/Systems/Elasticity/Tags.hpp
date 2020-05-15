// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving elasticity problems
 *
 * \details In elasticity problems we solve for the displacement vector
 * field \f$\boldsymbol{\xi}\f$ in an elastic material that responds to external
 * forces, stresses or deformations. In this static approximation the
 * equations of motion reduce to the elliptic equations
 *
 * \f[
 * \nabla_i T^{ij} = f_\mathrm{ext}^j
 * \f]
 *
 * that describes a state of equilibrium between the stresses \f$T^{ij}\f$
 * within the material and the external body forces
 * \f$\boldsymbol{f}_\mathrm{ext}\f$ (Eqns. 11.13 and 11.14 in
 * \cite ThorneBlandford2017 with the counteracting internal forces
 * \f$\boldsymbol{f} = -\boldsymbol{f}_\mathrm{ext}\f$). For small
 * deformations (see e.g. \cite ThorneBlandford2017, Section 11.3.2 for a
 * discussion) the stress is related to the strain
 * \f$S_{ij}=\nabla_{(i}\xi_{j)}\f$ by a linear constitutive relation
 * \f$T^{ij}=-Y^{ijkl}S_{kl}\f$ (Eq. 11.17 in \cite ThorneBlandford2017) that
 * describes the elastic properties of the material (see
 * `Elasticity::ConstitutiveRelations::ConstitutiveRelation`).
 */
namespace Elasticity::Tags {

/*!
 * \brief The material displacement field \f$\boldsymbol{u}(x)\f$
 */
template <size_t Dim>
struct Displacement : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

/*!
 * \brief The symmetric strain \f$S_{ij}=\nabla_{(i}u_{j)}\f$, describing the
 * deformation of the elastic material.
 */
template <size_t Dim>
struct Strain : db::SimpleTag {
  using type = tnsr::ii<DataVector, Dim>;
};

/*!
 * \brief The symmetric stress \f$T^{ij}\f$, describing pressure within the
 * elastic material.
 */
template <size_t Dim>
struct Stress : db::SimpleTag {
  using type = tnsr::II<DataVector, Dim>;
};

/*!
 * \brief The energy density \f$U\f$ stored in the deformation of the
 * elastic material.
 */
template <size_t Dim>
struct PotentialEnergy : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename FluxesComputerType, typename SolutionType>
struct FluxesComputer : elliptic::Tags::FluxesComputer<FluxesComputerType> {
  using type = FluxesComputerType;
  static std::string name() noexcept {
    return pretty_type::short_name<FluxesComputerType>();
  }
  using option_tags = tmpl::list<::OptionTags::AnalyticSolution<SolutionType>>;

  static constexpr bool pass_metavariables = false;
  static FluxesComputerType create_from_options(const SolutionType& solution) {
    return FluxesComputerType{
        std::make_unique<typename SolutionType::constitutive_relation_type>(
            solution.constitutive_relation())};
  }
};

}  // namespace Elasticity::Tags
