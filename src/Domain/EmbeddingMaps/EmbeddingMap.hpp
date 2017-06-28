// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the abstract base class EmbeddingMap.

#pragma once

#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"

/// Contains all embedding maps.
namespace EmbeddingMaps {}

/// \ingroup ComputationalDomain
/// Base class representing a coordinate mapping from a
/// `DimOfDomain`-dimensional space (coordinates \f$\xi^I\f$) to a
/// `DimOfRange`-dimensional space (coordinates \f$x^i\f$).
template <size_t DimOfDomain, size_t DimOfRange>
class EmbeddingMap : public PUP::able {
  static_assert(DimOfRange >= DimOfDomain,
                "EmbeddingMap DimOfDomain > DimOfRange");

 public:
  WRAPPED_PUPable_abstract(EmbeddingMap);  // NOLINT

  EmbeddingMap() = default;
  ~EmbeddingMap() override = default;
  EmbeddingMap(const EmbeddingMap<DimOfDomain, DimOfRange>&) = delete;
  EmbeddingMap(EmbeddingMap<DimOfDomain,  // NOLINT
                            DimOfRange>&&) noexcept = default;
  EmbeddingMap<DimOfDomain, DimOfRange>& operator=(
      const EmbeddingMap<DimOfDomain, DimOfRange>&) = delete;
  EmbeddingMap<DimOfDomain, DimOfRange>& operator=(
      EmbeddingMap<DimOfDomain, DimOfRange>&&) noexcept = default;

  /// Return a copy of the concrete EmbeddingMap.
  virtual std::unique_ptr<EmbeddingMap<DimOfDomain, DimOfRange>> get_clone()
      const = 0;

  /// Apply the mapping (i.e. compute \f$x^i(\xi^I)\f$).
  ///
  /// \return the point \f$x^i\f$ in the range of the mapping.
  /// \param xi the point \f$\xi^I\f$ in the domain of the mapping.
  virtual Point<DimOfRange, Frame::Grid> operator()(
      const Point<DimOfDomain, Frame::Logical>& xi) const = 0;

  /// Apply the inverse mapping (i.e. compute \f$\xi^I(x^i)\f$).
  ///
  /// \return the point \f$\xi^I\f$ in the domain of the mapping.
  /// \param x the point \f$x^i\f$ in the range of the mapping.
  virtual Point<DimOfDomain, Frame::Logical> inverse(
      const Point<DimOfRange, Frame::Grid>& x) const = 0;

  /*!
   * Compute a component of the Jacobian at the given domain point
   * (i.e. compute \f$\frac{\partial x^{ud}}{\partial \xi^{ld}}\f$).
   * \param xi the point \f$\Xi^I\f$ in the domain of the mapping.
   * \param ud the upper index of the Jacobian (a range index).
   * \param ld the lower index of the Jacobian (a domain index).
   */
  virtual double jacobian(const Point<DimOfDomain, Frame::Logical>& xi,
                          size_t ud, size_t ld) const = 0;

  /// (i.e. compute \f$\frac{\partial \xi^{ud}}{\partial x^{ld}}\f$).
  ///
  /// \param xi the point \f$\xi^I\f$ in the domain of the mapping.
  /// \param ud the upper index of the inverse Jacobian (a domain index).
  /// \param ld the lower index of the inverse Jacobian (a range index).
  virtual double inv_jacobian(const Point<DimOfDomain, Frame::Logical>& xi,
                              size_t ud, size_t ld) const = 0;
};
