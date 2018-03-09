// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

class DataVector;

namespace RelativisticEuler {
namespace Valencia {
struct TildeD : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "TildeD";
};

struct TildeTau : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "TildeTau";
};

template <size_t Dim>
struct TildeS : db::DataBoxTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static constexpr db::DataBoxString_t label = "TildeS";
};
}  // namespace Valencia
}  // namespace RelativisticEuler
