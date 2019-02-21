// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"

namespace NewtonianEuler {
/// %Tags for the conservative formulation of the Newtonian Euler system
namespace Tags {

/// The mass density of the fluid.
template <typename DataType>
struct MassDensity : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "MassDensity"; }
};

/// The momentum density of the fluid.
template <typename DataType, size_t Dim, typename Fr>
struct MomentumDensity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "MomentumDensity";
  }
};

/// The energy density of the fluid.
template <typename DataType>
struct EnergyDensity : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "EnergyDensity"; }
};

/// The macroscopic or flow velocity of the fluid.
template <typename DataType, size_t Dim, typename Fr>
struct Velocity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "Velocity";
  }
};

/// The specific internal energy of the fluid.
template <typename DataType>
struct SpecificInternalEnergy : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "SpecificInternalEnergy"; }
};

/// The fluid pressure.
template <typename DataType>
struct Pressure : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "Pressure"; }
};

}  // namespace Tags
}  // namespace NewtonianEuler
