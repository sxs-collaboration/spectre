// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Tags.hpp"
#include "Options/String.hpp"

namespace ForceFree {

/*!
 * \brief Tags for the GRFFE system with divergence cleaning.
 */
namespace Tags {

/*!
 * \brief The electric field \f$E^i\f$.
 */
struct ElectricField : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};

/*!
 * \brief The magnetic field \f$B^i\f$.
 */
struct MagneticField : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};

/*!
 * \brief The electric charge density \f$q\equiv-n_\mu J^\mu\f$ where
 * \f$n^\mu\f$ is the normal to hypersurface and \f$J^\mu\f$ is the 4-current.
 */
struct ChargeDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The spatial electric current density \f$J^i\f$.
 */
struct ElectricCurrentDensity : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};

/*!
 * \brief The divergence cleaning scalar field \f$\psi\f$ coupled to the
 * electric field.
 */
struct ElectricDivergenceCleaningField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The divergence cleaning scalar field \f$\phi\f$ coupled to the
 * magnetic field.
 */
struct MagneticDivergenceCleaningField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The densitized electric field \f$\tilde{E}^i = \sqrt{\gamma}E^i\f$.
 */
struct TildeE : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};

/*!
 * \brief The densitized magnetic field \f$\tilde{B}^i = \sqrt{\gamma}B^i\f$.
 */
struct TildeB : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};

/*!
 * \brief The densitized divergence cleaning field \f$\tilde{\psi} =
 * \sqrt{\gamma}\psi\f$ associated with the electric field.
 */
struct TildePsi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The densitized divergence cleaning field \f$\tilde{\phi} =
 * \sqrt{\gamma}\phi\f$ associated with the magnetic field.
 */
struct TildePhi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The densitized electric charge density \f$\tilde{q} =
 * \sqrt{\gamma}q\f$.
 */
struct TildeQ : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The densitized electric current density \f$\tilde{J}^i =
 * \alpha\sqrt{\gamma}J^i\f$.
 */
struct TildeJ : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};

/*!
 * \brief The largest characteristic speed
 */
struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};
}  // namespace Tags

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags related to the GRFFE evolution system.
 */
struct ForceFreeGroup {
  static std::string name() { return "ForceFree"; }
  static constexpr Options::String help{
      "Options for the GRFFE evolution system"};
  using group = evolution::OptionTags::SystemGroup;
};

/*!
 * \brief Groups option tags related to the divergence cleaning of the GRFFE
 * system.
 */
struct ConstraintDampingGroup {
  static std::string name() { return "ConstraintDamping"; }
  static constexpr Options::String help{
      "Options related to constraint damping"};
  using group = ForceFreeGroup;
};

/*!
 * \brief The constraint damping parameter for divergence cleaning of electric
 * fields.
 */
struct KappaPsi {
  static std::string name() { return "KappaPsi"; }
  using type = double;
  static constexpr Options::String help{
      "Constraint damping parameter for divergence cleaning of electric "
      "fields"};
  using group = ConstraintDampingGroup;
};

/*!
 * \brief The constraint damping parameter for divergence cleaning of magnetic
 * fields.
 */
struct KappaPhi {
  static std::string name() { return "KappaPhi"; }
  using type = double;
  static constexpr Options::String help{
      "Constraint damping parameter for divergence cleaning of magnetic "
      "fields"};
  using group = ConstraintDampingGroup;
};

/*!
 * \brief Groups option tags related to the electric current of the GRFFE
 * system.
 */
struct ForceFreeCurrentGroup {
  static std::string name() { return "ForceFreeCurrent"; }
  static constexpr Options::String help{
      "Options related to specifying the force-free electric current"};
  using group = ForceFreeGroup;
};

/*!
 * \brief The damping parameter in the electric current density to impose
 * force-free conditions. Physically, this parameter is the conductivity
 * parallel to magnetic field lines.
 */
struct ParallelConductivity {
  static std::string name() { return "ParallelConductivity"; }
  using type = double;
  static constexpr Options::String help{
      "Damping parameter for J^i to impose the force-free conditions, which is "
      "physically the conductivity parallel to B field"};
  using group = ForceFreeCurrentGroup;
};

}  // namespace OptionTags

namespace Tags {
/*!
 * \brief The constraint damping parameter \f$\kappa_\psi\f$ for divergence
 * cleaning of electric fields.
 */
struct KappaPsi : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::KappaPsi>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double kappa_psi) {
    return kappa_psi;
  }
};

/*!
 * \brief The constraint damping parameter \f$\kappa_\phi\f$ for divergence
 * cleaning of magnetic fields.
 */
struct KappaPhi : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::KappaPhi>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double kappa_phi) {
    return kappa_phi;
  }
};

/*!
 * \brief The damping parameter \f$\eta\f$ in the electric current density to
 * impose force-free conditions. Physically, this parameter is the conductivity
 * parallel to magnetic field lines.
 */
struct ParallelConductivity : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::ParallelConductivity>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double parallel_conductivity) {
    return parallel_conductivity;
  }
};

}  // namespace Tags

}  // namespace ForceFree
