// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"
#include "IO/Observer/Actions/GetLockPointer.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace CurvedScalarWave::Tags {
struct Psi;
struct Pi;
template <size_t SpatialDim>
struct Phi;
}  // namespace CurvedScalarWave::Tags
namespace Parallel {
class NodeLock;
}  // namespace Parallel
namespace intrp::OptionHolders {
struct Sphere;
}  // namespace intrp::OptionHolders
namespace intrp::Tags {
template <typename InterpolationTargetTag>
struct Sphere;
}  // namespace intrp::Tags
namespace intrp::TargetPoints {
template <typename InterpolationTargetTag, typename Frame>
struct Sphere;
}  // namespace intrp::TargetPoints
namespace observers::Tags {
struct H5FileLock;
}  // namespace observers::Tags
/// \endcond

namespace intrp::callbacks {
namespace DumpBondiSachsOnWorldtube_detail {
template <bool IncludeKleinGordon>
void apply_impl(
    gsl::not_null<Parallel::NodeLock*> hdf5_lock, double time,
    const OptionHolders::Sphere& sphere, const std::string& filename_prefix,
    const tnsr::aa<DataVector, 3, ::Frame::Inertial>& all_spacetime_metric,
    const tnsr::aa<DataVector, 3, ::Frame::Inertial>& all_pi,
    const tnsr::iaa<DataVector, 3, ::Frame::Inertial>& all_phi,
    const Scalar<DataVector>& all_csw_psi = {},
    const Scalar<DataVector>& all_csw_pi = {},
    const tnsr::i<DataVector, 3, ::Frame::Inertial>& all_csw_phi = {},
    const Scalar<DataVector>& all_lapse = {},
    const tnsr::I<DataVector, 3, ::Frame::Inertial>& all_shift = {});
}  // namespace DumpBondiSachsOnWorldtube_detail

/*!
 * \brief Post interpolation callback that dumps metric data in Bondi-Sachs form
 * on a number of extraction radii given by the `intrp::TargetPoints::Sphere`
 * target.
 *
 * To use this callback, the target must be the `intrp::TargetPoints::Sphere`
 * target in the inertial frame. This callback also expects that the GH source
 * vars on each of the target spheres are:
 *
 * - `gr::Tags::SpacetimeMetric`
 * - `gh::Tags::Pi`
 * - `gh::Tags::Phi`
 *
 * If IncludeKleinGordon is true, also expect:
 * - `CurvedScalarWave::Tags::Psi`
 * - `CurvedScalarWave::Tags::Pi`
 * - `CurvedScalarWave::Tags::Phi`
 * - `gr::Tags::Lapse`
 * - `gr::Tags::Shift`
 *
 * This callback will write a new `H5` file for each extraction radius in the
 * Sphere target. The name of this file will be a file prefix specified by the
 * Cce::Tags::FilePrefix prepended onto `CceRXXXX.h5` where the XXXX is the
 * zero-padded extraction radius rounded to the nearest integer. The quantities
 * that will be written are
 *
 * - `Cce::Tags::BondiBeta`
 * - `Cce::Tags::Dr<Cce::Tags::BondiJ>`
 * - `Cce::Tags::Du<Cce::Tags::BondiR>`
 * - `Cce::Tags::BondiH`
 * - `Cce::Tags::BondiJ`
 * - `Cce::Tags::BondiQ`
 * - `Cce::Tags::BondiR`
 * - `Cce::Tags::BondiU`
 * - `Cce::Tags::BondiW`
 *
 * If IncludeKleinGordon is true, also writes:
 * - `Cce::Tags::KleinGordonPsi`
 * - `Cce::Tags::KleinGordonPi`
 *
 * \note For all real quantities (Beta, DuR, R, W) (as well as the Klein-Gordon
 * Psi an Pi if included) we omit writing the negative m modes, and the
 * imaginary part of the m = 0 mode.
 */
template <typename InterpolationTargetTag, bool IncludeKleinGordon = false>
struct DumpBondiSachsOnWorldtube
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  static constexpr bool include_klein_gordon = IncludeKleinGordon;
  static constexpr double fill_invalid_points_with =
      std::numeric_limits<double>::quiet_NaN();

  using const_global_cache_tags = tmpl::list<Cce::Tags::FilePrefix>;

  using cce_boundary_tags = Cce::Tags::characteristic_worldtube_boundary_tags<
      Cce::Tags::BoundaryValue, include_klein_gordon>;

  using extra_klein_gordon_cce_tags =
      tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<3>, gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, 3>>;

  using gh_source_vars_for_cce = tmpl::append<
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>,
      tmpl::conditional_t<include_klein_gordon, extra_klein_gordon_cce_tags,
                          tmpl::list<>>>;

  using gh_source_vars_from_interpolation =
      typename InterpolationTargetTag::vars_to_interpolate_to_target;

  static_assert(
      tmpl::and_<
          std::is_same<tmpl::list_difference<gh_source_vars_from_interpolation,
                                             gh_source_vars_for_cce>,
                       tmpl::list<>>,
          std::is_same<tmpl::list_difference<gh_source_vars_for_cce,
                                             gh_source_vars_from_interpolation>,
                       tmpl::list<>>>::type::value,
      "To use DumpBondiSachsOnWorldtube, the GH source variables must be the "
      "spacetime metric, pi, and phi. If Klein Gordon variables are included, "
      "the source variables must also include the CurvedScalarWave Psi and Pi, "
      "as well as Lapse and shift.");

  static_assert(
      std::is_same_v<typename InterpolationTargetTag::compute_target_points,
                     intrp::TargetPoints::Sphere<InterpolationTargetTag,
                                                 ::Frame::Inertial>>,
      "To use the DumpBondiSachsOnWorldtube post interpolation callback, you "
      "must use the intrp::TargetPoints::Sphere target in the inertial "
      "frame");

  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    // Even though no other cores should be writing to this file, we
    // still need to get the h5 file lock so the system hdf5 doesn't get
    // upset
    auto* hdf5_lock = Parallel::local_synchronous_action<
        observers::Actions::GetLockPointer<observers::Tags::H5FileLock>>(
        Parallel::get_parallel_component<
            observers::ObserverWriter<Metavariables>>(cache));

    const double time =
        intrp::InterpolationTarget_detail::get_temporal_id_value(temporal_id);
    const auto& sphere =
        Parallel::get<Tags::Sphere<InterpolationTargetTag>>(cache);
    const auto& filename_prefix = Parallel::get<Cce::Tags::FilePrefix>(cache);
    const auto& all_spacetime_metric =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(box);
    const auto& all_pi = get<gh::Tags::Pi<DataVector, 3>>(box);
    const auto& all_phi = get<gh::Tags::Phi<DataVector, 3>>(box);
    if constexpr (IncludeKleinGordon) {
      const auto& all_csw_psi = get<CurvedScalarWave::Tags::Psi>(box);
      const auto& all_csw_pi = get<CurvedScalarWave::Tags::Pi>(box);
      const auto& all_csw_phi = get<CurvedScalarWave::Tags::Phi<3>>(box);
      const auto& all_lapse = get<gr::Tags::Lapse<DataVector>>(box);
      const auto& all_shift = get<gr::Tags::Shift<DataVector, 3>>(box);
      DumpBondiSachsOnWorldtube_detail::apply_impl<IncludeKleinGordon>(
          hdf5_lock, time, sphere, filename_prefix, all_spacetime_metric,
          all_pi, all_phi, all_csw_psi, all_csw_pi, all_csw_phi, all_lapse,
          all_shift);
    } else {
      DumpBondiSachsOnWorldtube_detail::apply_impl<IncludeKleinGordon>(
          hdf5_lock, time, sphere, filename_prefix, all_spacetime_metric,
          all_pi, all_phi);
    }
  }
};
}  // namespace intrp::callbacks
