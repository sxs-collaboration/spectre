// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/Source.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/String.hpp"

/// \cond
class DataVector;
/// \endcond

namespace NewtonianEuler {
/// %OptionTags for the conservative formulation of the Newtonian Euler system
namespace OptionTags {
template <size_t Dim>
struct SourceTerm {
  using type = std::unique_ptr<NewtonianEuler::Sources::Source<Dim>>;
  static constexpr Options::String help = "The source term to be used.";
  using group = ::evolution::OptionTags::SystemGroup;
};
}  // namespace OptionTags

/// %Tags for the conservative formulation of the Newtonian Euler system
namespace Tags {
/// The mass density of the fluid (as a conservative variable).
struct MassDensityCons : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The momentum density of the fluid.
template <size_t Dim, typename Fr>
struct MomentumDensity : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "MomentumDensity"; }
};

/// The energy density of the fluid.
struct EnergyDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The sound speed.
template <typename DataType>
struct SoundSpeed : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The characteristic speeds.
template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, Dim + 2>;
};

/// @{
/// The characteristic fields of the NewtonianEuler system.
struct VMinus : db::SimpleTag {
  using type = Scalar<DataVector>;
};
template <size_t Dim>
struct VMomentum : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};
struct VPlus : db::SimpleTag {
  using type = Scalar<DataVector>;
};
/// @}

/// The internal energy density.
template <typename DataType>
struct InternalEnergyDensity : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The kinetic energy density.
template <typename DataType>
struct KineticEnergyDensity : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The local Mach number of the flow
template <typename DataType>
struct MachNumber : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The ram pressure of the fluid.
template <typename DataType, size_t Dim, typename Fr>
struct RamPressure : db::SimpleTag {
  using type = tnsr::II<DataType, Dim, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "RamPressure"; }
};

/// The specific kinetic energy.
template <typename DataType>
struct SpecificKineticEnergy : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The source term in the evolution equations
template <size_t Dim>
struct SourceTerm : db::SimpleTag {
  using type = std::unique_ptr<NewtonianEuler::Sources::Source<Dim>>;

  using option_tags = tmpl::list<OptionTags::SourceTerm<Dim>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& source_term) {
    return source_term->get_clone();
  }
};

}  // namespace Tags
}  // namespace NewtonianEuler
