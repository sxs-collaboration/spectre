// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Identity.

#pragma once

#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/EmbeddingMaps/EmbeddingMap.hpp"
#include "Parallel/CharmPupable.hpp"

namespace EmbeddingMaps {

/// \ingroup EmbeddingMaps
/// Identity  map from \f$\xi \rightarrow x\f$.
template <size_t Dim>
class Identity : public EmbeddingMap<Dim, Dim> {
 public:
  Identity() = default;
  ~Identity() override = default;
  Identity(const Identity&) = delete;
  Identity(Identity&&) noexcept = default;  // NOLINT
  Identity& operator=(const Identity&) = delete;
  Identity& operator=(Identity&&) = default;

  Point<Dim, Frame::Grid> operator()(
      const Point<Dim, Frame::Logical>& xi) const override ;

  Point<Dim, Frame::Logical> inverse(
      const Point<Dim, Frame::Grid>& x) const override ;

  double jacobian(const Point<Dim, Frame::Logical>& /* xi */, const size_t ud,
                  const size_t ld) const override ;

  double inv_jacobian(const Point<Dim, Frame::Logical>& /* xi */,
                      const size_t ud, const size_t ld) const override ;

  std::unique_ptr<EmbeddingMap<Dim, Dim>> get_clone() const override ;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EmbeddingMap<Dim, Dim>), Identity);

  explicit Identity(CkMigrateMessage* /* m */) {}

  void pup(PUP::er& p) override;
};
}  // namespace EmbeddingMaps

/// \cond HIDDEN_SYMBOLS
template <size_t Dim>
PUP::able::PUP_ID EmbeddingMaps::Identity<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond
