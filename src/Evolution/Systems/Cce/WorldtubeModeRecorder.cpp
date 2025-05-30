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
#include "Utilities/ConstantExpressions.hpp"
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

template <>
std::string dataset_label_for_tag<Cce::Tags::KleinGordonPsi>() {
  return "KGPsi";
}

template <>
std::string dataset_label_for_tag<Cce::Tags::KleinGordonPi>() {
  return "dtKGPsi";
}

WorldtubeModeRecorder::WorldtubeModeRecorder() = default;
WorldtubeModeRecorder::WorldtubeModeRecorder(const size_t output_l_max,
                                             const std::string& h5_filename)
    : output_l_max_(output_l_max),
      output_file_(h5_filename, true),
      all_legend_(build_legend(false)),
      real_legend_(build_legend(true)),
      data_to_write_buffer_(data_to_write_size(false)),
      goldberg_mode_buffer_(square(output_l_max_ + 1)) {}

template <int Spin>
void WorldtubeModeRecorder::append_modal_data(
    const std::string& subfile_path, const double time,
    const ComplexDataVector& nodal_data, const size_t data_l_max) {
  check_data_l_max(data_l_max);
  // Set some views
  SpinWeighted<ComplexDataVector, Spin> nodal_data_view;
  nodal_data_view.set_data_ref(
      make_not_null(&const_cast<ComplexDataVector&>(nodal_data)));  // NOLINT
  const size_t number_of_data_modes = square(data_l_max + 1);
  if (goldberg_mode_buffer_.size() < number_of_data_modes) {
    goldberg_mode_buffer_.destructive_resize(number_of_data_modes);
  }
  SpinWeighted<ComplexModalVector, Spin> goldberg_modes;
  goldberg_modes.set_data_ref(goldberg_mode_buffer_.data(),
                              number_of_data_modes);

  // First transform to coefficients using swsh_transform, and then convert
  // libsharp coefficients into modes
  Spectral::Swsh::libsharp_to_goldberg_modes(
      make_not_null(&goldberg_modes),
      Spectral::Swsh::swsh_transform(data_l_max, 1, nodal_data_view),
      data_l_max);

  append_modal_data<Spin>(subfile_path, time, goldberg_modes.data(),
                          data_l_max);
}

template <int Spin>
void WorldtubeModeRecorder::append_modal_data(
    const std::string& subfile_path, const double time,
    const ComplexModalVector& modal_data, const size_t data_l_max) {
  constexpr bool is_real = Spin == 0;
  check_data_l_max(data_l_max);

  ASSERT(data_to_write_buffer_.capacity() == data_to_write_size(false),
         "Buffer does not have the correct capactiy. Was expecting "
             << data_to_write_size(false) << " but got "
             << data_to_write_buffer_.capacity());

  // This won't remove the allocation, only removes the elements so we don't
  // have to do complicated index tracking
  data_to_write_buffer_.clear();
  data_to_write_buffer_.push_back(time);

  // Because the goldberg format is strictly increasing l modes, to restrict, we
  // just take the first output_l_max_ modes.
  // NOLINTBEGIN
  const ComplexModalVector modal_data_view{
      const_cast<ComplexModalVector&>(modal_data).data(),
      square(output_l_max_ + 1)};
  // NOLINTEND

  // We loop over ell and m rather than just the total number of modes
  // because we don't print negative m or the imaginary part of m=0
  // for real quantities.
  for (size_t ell = 0; ell <= output_l_max_; ell++) {
    for (int m = is_real ? 0 : -static_cast<int>(ell);
         m <= static_cast<int>(ell); m++) {
      const size_t goldberg_index =
          Spectral::Swsh::goldberg_mode_index(output_l_max_, ell, m);
      data_to_write_buffer_.push_back(real(modal_data_view[goldberg_index]));
      if (not is_real or m != 0) {
        data_to_write_buffer_.push_back(imag(modal_data_view[goldberg_index]));
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
  return 1 + square(output_l_max_ + 1) * (is_real ? 1 : 2);
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
  for (int ell = 0; ell <= static_cast<int>(output_l_max_); ++ell) {
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

void WorldtubeModeRecorder::check_data_l_max(const size_t data_l_max) const {
  if (UNLIKELY(data_l_max < output_l_max_)) {
    ERROR(
        "WorldtubeModeRecorder can only do a restriction operation, not an "
        "elongation operation. Said another way, the LMax of data passed to "
        "WorldtubeModeRecorder ("
        << data_l_max
        << ") must be greater than or equal to the LMax the class was "
           "constructed with ("
        << output_l_max_ << ").");
  }
}

#define SPIN(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                          \
  template void WorldtubeModeRecorder::append_modal_data<SPIN(data)>( \
      const std::string& subfile_path, double time,                   \
      const ComplexDataVector& nodal_data, const size_t data_l_max);  \
  template void WorldtubeModeRecorder::append_modal_data<SPIN(data)>( \
      const std::string& subfile_path, double time,                   \
      const ComplexModalVector& modal_data, const size_t data_l_max);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))

#undef INSTANTIATE
#undef SPIN
}  // namespace Cce
