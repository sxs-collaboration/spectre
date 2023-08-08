// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// Boundary corrections/numerical fluxes
namespace ScalarTensor::BoundaryCorrections {
/// \cond
template <typename DerivedGhCorrection, typename DerivedScalarCorrection>
class ProductOfCorrections;
/// \endcond

namespace detail {

template <typename GhList, typename ScalarList>
struct AllProductCorrections;

template <typename GhList, typename... ScalarCorrections>
struct AllProductCorrections<GhList, tmpl::list<ScalarCorrections...>> {
  using type = tmpl::flatten<tmpl::list<
      tmpl::transform<GhList, tmpl::bind<ProductOfCorrections, tmpl::_1,
                                         tmpl::pin<ScalarCorrections>>>...>>;
};
}  // namespace detail

/*!
 * \brief The base class used to make boundary corrections factory creatable so
 * they can be specified in the input file.
 */
class BoundaryCorrection : public PUP::able {
 public:
  BoundaryCorrection() = default;
  BoundaryCorrection(const BoundaryCorrection&) = default;
  BoundaryCorrection& operator=(const BoundaryCorrection&) = default;
  BoundaryCorrection(BoundaryCorrection&&) = default;
  BoundaryCorrection& operator=(BoundaryCorrection&&) = default;
  ~BoundaryCorrection() override = default;

  /// \cond
  explicit BoundaryCorrection(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT
  /// \endcond

  using creatable_classes = typename detail::AllProductCorrections<
      typename gh::BoundaryCorrections::BoundaryCorrection<
          3>::creatable_classes,
      typename CurvedScalarWave::BoundaryCorrections::BoundaryCorrection<
          3>::creatable_classes>::type;

  virtual std::unique_ptr<BoundaryCorrection> get_clone() const = 0;
};
}  // namespace ScalarTensor::BoundaryCorrections
