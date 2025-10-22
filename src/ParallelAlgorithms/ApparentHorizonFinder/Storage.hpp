// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Parallel/MultiReaderSpinlock.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"

namespace ah::Storage {
/*!
 * \brief Holds the `ah::source_vars`, mesh, and other variables on the horizon
 * finder for a given element.
 */
template <typename Fr>
struct VolumeVariables {
  /*!
   * \brief The mesh that corresponds to the volume vars.
   */
  Mesh<3> mesh;

  /*!
   * \brief A `Variables` of the tensors in the volume that we need to
   * interpolate onto the horizon.
   */
  Variables<ah::vars_to_interpolate_to_target<3, Fr>>
      vars_to_interpolate_to_target{};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Fr>
bool operator==(const VolumeVariables<Fr>& lhs, const VolumeVariables<Fr>& rhs);
template <typename Fr>
bool operator!=(const VolumeVariables<Fr>& lhs, const VolumeVariables<Fr>& rhs);

/*!
 * \brief Holds the `ylm::Strahlkorper` and associated quantities for a single
 * `FastFlow` iteration.
 */
template <typename Fr>
struct Iteration {
  /*!
   * \brief The surface for this iteration
   */
  ylm::Strahlkorper<Fr> strahlkorper;
  /*!
   *  \brief Holds the list of all points (in block logical coordinates) that
   *  need to be interpolated onto.
   *
   * \details If an element of the vector is `std::nullopt`, then that point is
   * outside the domain.
   */
  std::optional<std::vector<BlockLogicalCoords<3>>> block_coord_holders;
  /*!
   * \brief Holds the interpolated `Variables` on the points in
   * `block_coord_holders`.
   *
   * \details The grid points inside are indexed according to
   * `block_coord_holders`.
   */
  Variables<ah::vars_to_interpolate_to_target<3, Fr>> interpolated_vars{};
  /*!
   * \brief Keeps track of the indices in `interpolated_vars` that have
   * already been interpolated to.
   */
  std::vector<bool> indices_interpolated_to_thus_far{};
  /*!
   * \brief Holds the element IDs of all elements that intersect with the
   * current iteration surface. Used to determine which elements will send
   * data for the next horizon find.
   */
  std::unordered_set<ElementId<3>> intersecting_element_ids{};
  /*!
   * \brief Offsets of newly interpolated points in the overall tensor (used as
   * memory buffer)
   */
  std::vector<size_t> offsets_of_newly_interpolated_points{};
  /*!
   * \brief Logical coordinates of newly interpolated points (used as memory
   * buffer)
   *
   * These `std::vector`s are used to reserve memory and then append points to
   * them as we find them in an element. The memory is reused for each element.
   * Then, a non-owning DataVector is created by pointing into this memory.
   * That's why this is a `std::array` of `std::vector`s, not vice versa.
   */
  std::array<std::vector<double>, 3>
      x_element_logical_of_newly_interpolated_points{};
  /*!
   * \brief Buffer for newly interpolated variables (used as memory buffer)
   */
  std::vector<double> newly_interpolated_vars_buffer{};

  /*!
   * \brief How many times we've tried to compute the coordinates for this
   * iteration.
   */
  size_t compute_coords_retries = 0;

  /*!
   * \brief Whether all points in `interpolated_vars` have been filled.
   */
  bool interpolation_is_complete() const;

  void reset_for_next_iteration();

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Fr>
bool operator==(const Iteration<Fr>& lhs, const Iteration<Fr>& rhs);
template <typename Fr>
bool operator!=(const Iteration<Fr>& lhs, const Iteration<Fr>& rhs);

/*!
 * \brief Holds all data necessary for a single horizon find.
 *
 * \details This includes volume variables which persist for the entire horizon
 * find, and also interpolated variables that are updated for each iteration.
 */
template <typename Fr>
struct SingleTimeStorage {
  /*!
   * \brief Map between `ElementId`s and the volume variables from that element.
   */
  std::unordered_map<ElementId<3>, VolumeVariables<Fr>> all_volume_variables;
  /*!
   * \brief Elements in which we have found points to interpolate to in previous
   * iterations, to try first before searching all elements.
   *
   * This is not only a performance optimization, but also important for
   * robustness. If we try to interpolate from elements in a different order in
   * each iteration, then points that lie directly on element boundaries can
   * fluctuate in interpolated value, preventing convergence (see
   * https://github.com/sxs-collaboration/spectre/issues/3899).
   */
  std::vector<ElementId<3>> element_order{};
  /*!
   * \brief The `Iteration` data for the current fast flow iteration.
   */
  Iteration<Fr> current_iteration;
  /*!
   * \brief The previous iteration surface, used if interpolation fails.
   */
  ylm::Strahlkorper<Fr> previous_iteration_surface;
  /*!
   * \brief The `ah::Destination` for this horizon find.
   */
  Destination destination{};
  /*!
   * \brief Whether we have checked if the functions of time are up to date for
   * this horizon find.
   */
  bool time_is_ready = false;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Fr>
bool operator==(const SingleTimeStorage<Fr>& lhs,
                const SingleTimeStorage<Fr>& rhs);
template <typename Fr>
bool operator!=(const SingleTimeStorage<Fr>& lhs,
                const SingleTimeStorage<Fr>& rhs);

/*!
 * \brief The time and final surface for a previous horizon find.
 */
template <typename Fr>
struct PreviousSurface {
  PreviousSurface() = default;
  PreviousSurface(const LinkedMessageId<double>& time_in,
                  ylm::Strahlkorper<Fr> surface_in,
                  std::unordered_set<ElementId<3>> intersecting_element_ids_in);

  LinkedMessageId<double> time;
  ylm::Strahlkorper<Fr> surface;
  std::unordered_set<ElementId<3>> intersecting_element_ids;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Fr>
bool operator==(const PreviousSurface<Fr>& lhs, const PreviousSurface<Fr>& rhs);
template <typename Fr>
bool operator!=(const PreviousSurface<Fr>& lhs, const PreviousSurface<Fr>& rhs);

/*!
 * \brief Holds a previous surface and a lock that protects it.
 *
 * \details This is used to store and update a previous surface in the global
 * cache and allow multiple readers (elements) to access it simultaneously.
 */
template <typename Fr>
struct LockedPreviousSurface {
  std::optional<PreviousSurface<Fr>> surface;
  // Lock is mutable so it can be retrieved from the const global cache and put
  // in read-lock mode by elements.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Parallel::MultiReaderSpinlock lock;

  LockedPreviousSurface();
  explicit LockedPreviousSurface(const PreviousSurface<Fr>& rhs);
  LockedPreviousSurface(const LockedPreviousSurface& rhs);
  LockedPreviousSurface& operator=(const LockedPreviousSurface& rhs);
  LockedPreviousSurface(LockedPreviousSurface&& rhs);
  LockedPreviousSurface& operator=(LockedPreviousSurface&& rhs);
  ~LockedPreviousSurface() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Fr>
bool operator==(const LockedPreviousSurface<Fr>& lhs,
                const LockedPreviousSurface<Fr>& rhs);
template <typename Fr>
bool operator!=(const LockedPreviousSurface<Fr>& lhs,
                const LockedPreviousSurface<Fr>& rhs);

}  // namespace ah::Storage
