// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "Evolution/TypeTraits.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"

/// \cond
class DataVector;
/// \endcond

namespace NewtonianEuler {
/// %Tags for the conservative formulation of the Newtonian Euler system
namespace Tags {

/// The mass density of the fluid.
template <typename DataType>
struct MassDensity : db::SimpleTag {
  using type = Scalar<DataType>;
};

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

/// The macroscopic or flow velocity of the fluid.
template <typename DataType, size_t Dim, typename Fr>
struct Velocity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "Velocity"; }
};

/// The specific internal energy of the fluid.
template <typename DataType>
struct SpecificInternalEnergy : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The fluid pressure.
template <typename DataType>
struct Pressure : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The sound speed.
template <typename DataType>
struct SoundSpeed : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The square of the sound speed.
template <typename DataType>
struct SoundSpeedSquared : db::SimpleTag {
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

/// Base tag for the source term
struct SourceTermBase : db::BaseTag {};

/// The source term in the evolution equations
template <typename InitialDataType>
struct SourceTerm : SourceTermBase, db::SimpleTag {
  using type = typename InitialDataType::source_term_type;
  using option_tags = tmpl::list<
      tmpl::conditional_t<evolution::is_analytic_solution_v<InitialDataType>,
                          ::OptionTags::AnalyticSolution<InitialDataType>,
                          ::OptionTags::AnalyticData<InitialDataType>>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const InitialDataType& initial_data) {
    return initial_data.source_term();
  }
};

}  // namespace Tags
}  // namespace NewtonianEuler
