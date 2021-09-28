// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// Boundary corrections/numerical fluxes
namespace grmhd::GhValenciaDivClean::BoundaryCorrections {
/// \cond
template <typename DerivedGhCorrection, typename DerivedValenciaCorrection>
class ProductOfCorrections;
/// \endcond

namespace detail {
template <typename GhList, typename ValenciaList>
struct AllProductCorrections;

template <typename GhList, typename... ValenciaCorrections>
struct AllProductCorrections<GhList, tmpl::list<ValenciaCorrections...>> {
  using type = tmpl::flatten<tmpl::list<
      tmpl::transform<GhList, tmpl::bind<ProductOfCorrections, tmpl::_1,
                                         tmpl::pin<ValenciaCorrections>>>...>>;
};
}  // namespace detail

/*!
 * \brief The base class used to make boundary corrections factory createable so
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
      typename GeneralizedHarmonic::BoundaryCorrections::BoundaryCorrection<
          3_st>::creatable_classes,
      typename grmhd::ValenciaDivClean::BoundaryCorrections::
          BoundaryCorrection::creatable_classes>::type;

  virtual std::unique_ptr<BoundaryCorrection> get_clone() const = 0;
};
}  // namespace grmhd::GhValenciaDivClean::BoundaryCorrections
