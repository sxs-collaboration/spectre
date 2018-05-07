// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

namespace NewtonianEuler {
namespace Tags {

/// The mass density of the fluid.
template <typename DataType>
struct MassDensity : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "MassDensity"; }
};

/// The momentum density of the fluid.
template <typename DataType, size_t Dim, typename VolumeFrame = Frame::Inertial>
struct MomentumDensity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, VolumeFrame>;
  static std::string name() noexcept { return "MomentumDensity"; }
};

/// The energy density of the fluid.
template <typename DataType>
struct EnergyDensity : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "EnergyDensity"; }
};

/// The macroscopic or flow velocity of the fluid.
template <typename DataType, size_t Dim, typename VolumeFrame = Frame::Inertial>
struct Velocity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, VolumeFrame>;
  static std::string name() noexcept { return "Velocity"; }
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
