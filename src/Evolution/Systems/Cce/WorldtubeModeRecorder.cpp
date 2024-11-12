// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
template <>
std::string dataset_label_for_tag<Cce::Tags::BondiBeta>() {
  return "Beta";
}

template <>
std::string dataset_label_for_tag<Cce::Tags::BondiU>() {
  return "U";
}

template <>
std::string dataset_label_for_tag<Cce::Tags::BondiQ>() {
  return "Q";
}

template <>
std::string dataset_label_for_tag<Cce::Tags::BondiW>() {
  return "W";
}

template <>
std::string dataset_label_for_tag<Cce::Tags::BondiJ>() {
  return "J";
}

template <>
std::string dataset_label_for_tag<Cce::Tags::Dr<Cce::Tags::BondiJ>>() {
  return "DrJ";
}

template <>
std::string dataset_label_for_tag<Cce::Tags::Du<Cce::Tags::BondiJ>>() {
  return "H";
}

template <>
std::string dataset_label_for_tag<Cce::Tags::BondiR>() {
  return "R";
}

template <>
std::string dataset_label_for_tag<Cce::Tags::Du<Cce::Tags::BondiR>>() {
  return "DuR";
}

WorldtubeModeRecorder::WorldtubeModeRecorder() = default;
WorldtubeModeRecorder::WorldtubeModeRecorder(const size_t l_max,
                                             const std::string& h5_filename)
    : l_max_(l_max),
      output_file_(h5_filename, true),
      all_legend_(build_legend(false)),
      real_legend_(build_legend(true)),
      data_to_write_buffer_(data_to_write_size(false)),
      goldberg_mode_buffer_(square(l_max_ + 1)) {}

template <int Spin>
void WorldtubeModeRecorder::append_modal_data(
    const std::string& subfile_path, const double time,
    const ComplexDataVector& nodal_data) {
  // Set some views
  SpinWeighted<ComplexDataVector, Spin> nodal_data_view;
  nodal_data_view.set_data_ref(
      make_not_null(&const_cast<ComplexDataVector&>(nodal_data)));  // NOLINT
  SpinWeighted<ComplexModalVector, Spin> goldberg_modes;
  goldberg_modes.set_data_ref(make_not_null(&goldberg_mode_buffer_));

  // First transform to coefficients using swsh_transform, and then convert
  // libsharp coefficients into modes
  Spectral::Swsh::libsharp_to_goldberg_modes(
      make_not_null(&goldberg_modes),
      Spectral::Swsh::swsh_transform(l_max_, 1, nodal_data_view), l_max_);

  append_modal_data<Spin>(subfile_path, time, goldberg_modes.data());
}

template <int Spin>
void WorldtubeModeRecorder::append_modal_data(
    const std::string& subfile_path, const double time,
    const ComplexModalVector& modal_data) {
  constexpr bool is_real = Spin == 0;

  ASSERT(data_to_write_buffer_.capacity() == data_to_write_size(false),
         "Buffer does not have the correct capactiy. Was expecting "
             << data_to_write_size(false) << " but got "
             << data_to_write_buffer_.capacity());

  // This won't remove the allocation, only removes the elements so we don't
  // have to do complicated index tracking
  data_to_write_buffer_.clear();
  data_to_write_buffer_.push_back(time);

  // We loop over ell and m rather than just the total number of modes
  // because we don't print negative m or the imaginary part of m=0
  // for real quantities.
  for (size_t ell = 0; ell <= l_max_; ell++) {
    for (int m = is_real ? 0 : -static_cast<int>(ell);
         m <= static_cast<int>(ell); m++) {
      const size_t goldberg_index =
          Spectral::Swsh::goldberg_mode_index(l_max_, ell, m);
      data_to_write_buffer_.push_back(real(modal_data[goldberg_index]));
      if (not is_real or m != 0) {
        data_to_write_buffer_.push_back(imag(modal_data[goldberg_index]));
      }
    }
  }

  // Sanity check
  ASSERT(data_to_write_buffer_.size() == data_to_write_size(is_real),
         "Buffer does not have the correct size. Was expecting "
             << data_to_write_size(is_real) << " but got "
             << data_to_write_buffer_.size());

  const std::vector<std::string>& legend =
      is_real ? real_legend() : all_legend();
  auto& output_mode_dataset =
      output_file_.try_insert<h5::Dat>(subfile_path, legend, 0);
  output_mode_dataset.append(data_to_write_buffer_);
  output_file_.close_current_object();
}

size_t WorldtubeModeRecorder::data_to_write_size(const bool is_real) const {
  return 1 + square(l_max_ + 1) * (is_real ? 1 : 2);
}

const std::vector<std::string>& WorldtubeModeRecorder::all_legend() const {
  return all_legend_;
}
const std::vector<std::string>& WorldtubeModeRecorder::real_legend() const {
  return real_legend_;
}

std::vector<std::string> WorldtubeModeRecorder::build_legend(
    const bool is_real) const {
  std::vector<std::string> legend;
  legend.reserve(data_to_write_size(is_real));
  legend.emplace_back("Time");
  for (int ell = 0; ell <= static_cast<int>(l_max_); ++ell) {
    for (int m = is_real ? 0 : -ell; m <= ell; ++m) {
      legend.push_back(MakeString{} << "Re(" << ell << "," << m << ")");
      // For real quantities, don't include the imaginary m=0
      if (not is_real or m != 0) {
        legend.push_back(MakeString{} << "Im(" << ell << "," << m << ")");
      }
    }
  }
  return legend;
}

#define SPIN(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                          \
  template void WorldtubeModeRecorder::append_modal_data<SPIN(data)>( \
      const std::string& subfile_path, double time,                   \
      const ComplexDataVector& nodal_data);                           \
  template void WorldtubeModeRecorder::append_modal_data<SPIN(data)>( \
      const std::string& subfile_path, double time,                   \
      const ComplexModalVector& modal_data);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))

#undef INSTANTIATE
#undef SPIN
}  // namespace Cce
