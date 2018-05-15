// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for scalar wave system

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

class DataVector;

namespace ScalarWave {
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Psi"; }
  static constexpr bool should_be_sliced_to_boundary = true;
};

struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Pi"; }
  static constexpr bool should_be_sliced_to_boundary = true;
};

template <size_t Dim>
struct Phi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static std::string name() noexcept { return "Phi"; }
  static constexpr bool should_be_sliced_to_boundary = true;
};
}  // namespace ScalarWave
