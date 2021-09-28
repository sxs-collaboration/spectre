// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/DgSubcell/TciStatus.hpp"

/// \cond
class DataVector;
namespace domain::Tags {
template <size_t Dim>
struct Mesh;
}  // namespace domain::Tags
namespace evolution::dg::subcell::Tags {
struct ActiveGrid;
template <size_t Dim>
struct Mesh;
struct TciGridHistory;
}  // namespace evolution::dg::subcell::Tags
/// \endcond

namespace evolution::dg::subcell::Tags {
/// Stores the status of the troubled cell indicator in the element as a
/// `Scalar<DataVector>` so it can be observed.
struct TciStatus : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Computes the TCI status from the currently active grid and the TCI history.
template <size_t Dim>
struct TciStatusCompute : TciStatus, db::ComputeTag {
  using base = TciStatus;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<::domain::Tags::Mesh<Dim>, Mesh<Dim>,
                                   ActiveGrid, TciGridHistory>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const ::Mesh<Dim>&,
      const ::Mesh<Dim>&, evolution::dg::subcell::ActiveGrid,
      const std::deque<evolution::dg::subcell::ActiveGrid>&)>(
      &evolution::dg::subcell::tci_status<Dim>);
};
}  // namespace evolution::dg::subcell::Tags
