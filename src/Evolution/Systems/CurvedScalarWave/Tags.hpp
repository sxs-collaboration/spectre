// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the curved scalar wave system

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

class DataVector;

namespace CurvedScalarWave {
struct Psi : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Psi";
};

struct Pi : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Pi";
};

template <size_t Dim>
struct Phi : db::DataBoxTag {
  using type = tnsr::i<DataVector, Dim>;
  static constexpr db::DataBoxString label = "Phi";
};

struct ConstraintGamma1 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "ConstraintGamma1";
};

struct ConstraintGamma2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "ConstraintGamma2";
};
}  // namespace CurvedScalarWave
