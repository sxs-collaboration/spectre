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
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static constexpr db::Label label = "Psi";
};

struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static constexpr db::Label label = "Pi";
};

template <size_t Dim>
struct Phi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
  static constexpr db::Label label = "Phi";
};

struct ConstraintGamma1 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static constexpr db::Label label = "ConstraintGamma1";
};

struct ConstraintGamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static constexpr db::Label label = "ConstraintGamma2";
};
}  // namespace CurvedScalarWave
