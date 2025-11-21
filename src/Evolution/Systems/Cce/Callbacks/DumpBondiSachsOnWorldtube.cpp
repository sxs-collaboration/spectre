// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Callbacks/DumpBondiSachsOnWorldtube.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <mutex>
#include <string>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/NodeLock.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp::callbacks::DumpBondiSachsOnWorldtube_detail {
template <bool IncludeKleinGordon>
void apply_impl(
    const gsl::not_null<Parallel::NodeLock*> hdf5_lock, const double time,
    const OptionHolders::Sphere& sphere, const std::string& filename_prefix,
    const tnsr::aa<DataVector, 3, ::Frame::Inertial>& all_spacetime_metric,
    const tnsr::aa<DataVector, 3, ::Frame::Inertial>& all_pi,
    const tnsr::iaa<DataVector, 3, ::Frame::Inertial>& all_phi,
    const Scalar<DataVector>& all_csw_psi, const Scalar<DataVector>& all_csw_pi,
    const tnsr::i<DataVector, 3, ::Frame::Inertial>& all_csw_phi,
    const Scalar<DataVector>& all_lapse,
    const tnsr::I<DataVector, 3, ::Frame::Inertial>& all_shift) {
  using cce_boundary_tags = Cce::Tags::characteristic_worldtube_boundary_tags<
      Cce::Tags::BoundaryValue, IncludeKleinGordon>;
  using tags_for_writing =
      Cce::Tags::worldtube_boundary_tags_for_writing<Cce::Tags::BoundaryValue,
                                                     IncludeKleinGordon>;
  static_assert(
      std::is_same_v<tmpl::list_difference<tags_for_writing, cce_boundary_tags>,
                     tmpl::list<>>,
      "Cce tags to dump are not in the boundary tags.");

  if (sphere.angular_ordering != ylm::AngularOrdering::Cce) {
    ERROR(
        "To use the DumpBondiSachsOnWorldtube post interpolation callback, "
        "the angular ordering of the Spheres must be Cce, not "
        << sphere.angular_ordering);
  }

  const auto& radii = sphere.radii;
  const size_t l_max = sphere.l_max;
  const size_t num_points_single_sphere =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Bondi data
  Variables<cce_boundary_tags> bondi_boundary_data{num_points_single_sphere};

  size_t offset = 0;
  for (const auto& radius : radii) {
    const tnsr::aa<DataVector, 3, ::Frame::Inertial> spacetime_metric;
    const tnsr::aa<DataVector, 3, ::Frame::Inertial> pi;
    const tnsr::iaa<DataVector, 3, ::Frame::Inertial> phi;

    // Set data references so we don't copy data unnecessarily
    for (size_t a = 0; a < 4; a++) {
      for (size_t b = 0; b < 4; b++) {
        make_const_view(make_not_null(&spacetime_metric.get(a, b)),
                        all_spacetime_metric.get(a, b), offset,
                        num_points_single_sphere);
        make_const_view(make_not_null(&pi.get(a, b)), all_pi.get(a, b), offset,
                        num_points_single_sphere);
        for (size_t i = 0; i < 3; i++) {
          make_const_view(make_not_null(&phi.get(i, a, b)),
                          all_phi.get(i, a, b), offset,
                          num_points_single_sphere);
        }
      }
    }

    {
      auto non_klein_gordon_data =
          bondi_boundary_data.template reference_subset<
              Cce::Tags::characteristic_worldtube_boundary_tags<
                  Cce::Tags::BoundaryValue>>();
      Cce::create_bondi_boundary_data(make_not_null(&non_klein_gordon_data),
                                      phi, pi, spacetime_metric, radius, l_max);
    }

    if constexpr (IncludeKleinGordon) {
      const Scalar<DataVector> csw_psi;
      const Scalar<DataVector> csw_pi;
      const Scalar<DataVector> lapse;
      const tnsr::i<DataVector, 3, ::Frame::Inertial> csw_phi;
      const tnsr::I<DataVector, 3, ::Frame::Inertial> shift;

      make_const_view(make_not_null(&csw_psi.get()), all_csw_psi.get(), offset,
                      num_points_single_sphere);
      make_const_view(make_not_null(&csw_pi.get()), all_csw_pi.get(), offset,
                      num_points_single_sphere);
      make_const_view(make_not_null(&lapse.get()), all_lapse.get(), offset,
                      num_points_single_sphere);
      for (size_t i = 0; i < 3; i++) {
        make_const_view(make_not_null(&csw_phi.get(i)), all_csw_phi.get(i),
                        offset, num_points_single_sphere);
        make_const_view(make_not_null(&shift.get(i)), all_shift.get(i), offset,
                        num_points_single_sphere);
      }

      Cce::create_klein_gordon_boundary_data(
          make_not_null(&bondi_boundary_data), csw_phi, csw_pi, csw_psi, lapse,
          shift);
    }

    offset += num_points_single_sphere;

    const std::string filename =
        MakeString{} << filename_prefix << "CceR" << std::setfill('0')
                     << std::setw(4) << std::lround(radius) << ".h5";

    // Lock now and it'll be unlocked for this radius after we finish writing
    // the data to disk
    const std::lock_guard lock(*hdf5_lock);
    Cce::WorldtubeModeRecorder recorder{l_max, filename};

    tmpl::for_each<tags_for_writing>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          constexpr int spin = Tag::tag::type::type::spin;

          const ComplexDataVector& bondi_nodal_data =
              get(get<Tag>(bondi_boundary_data)).data();

          recorder.append_modal_data<spin>(
              Cce::dataset_label_for_tag<typename Tag::tag>(), time,
              bondi_nodal_data, l_max);
        });
  }
}

#define INCLUDE_KLEIN_GORDON(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void apply_impl<INCLUDE_KLEIN_GORDON(data)>(                        \
      const gsl::not_null<Parallel::NodeLock*> hdf5_lock, const double time,   \
      const OptionHolders::Sphere& sphere, const std::string& filename_prefix, \
      const tnsr::aa<DataVector, 3, ::Frame::Inertial>& all_spacetime_metric,  \
      const tnsr::aa<DataVector, 3, ::Frame::Inertial>& all_pi,                \
      const tnsr::iaa<DataVector, 3, ::Frame::Inertial>& all_phi,              \
      const Scalar<DataVector>& all_csw_psi,                                   \
      const Scalar<DataVector>& all_csw_pi,                                    \
      const tnsr::i<DataVector, 3, ::Frame::Inertial>& all_csw_phi,            \
      const Scalar<DataVector>& all_lapse,                                     \
      const tnsr::I<DataVector, 3, ::Frame::Inertial>& all_shift);

GENERATE_INSTANTIATIONS(INSTANTIATE, (true, false))

#undef INSTANTIATE
#undef INCLUDE_KLEIN_GORDON
}  // namespace intrp::callbacks::DumpBondiSachsOnWorldtube_detail
