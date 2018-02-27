// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for scalar wave system

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

class DataVector;

namespace ScalarWave {
struct Psi : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Psi";
  static constexpr bool should_be_sliced_to_boundary = true;
};

struct Pi : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Pi";
  static constexpr bool should_be_sliced_to_boundary = true;
};

template <size_t Dim>
struct Phi : db::DataBoxTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static constexpr db::DataBoxString label = "Phi";
  static constexpr bool should_be_sliced_to_boundary = true;
};
}  // namespace ScalarWave
