
// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

void puncture_field_kerr_0(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, const double bh_mass,
    const std::array<double, 3>& bh_spin) {
  const size_t grid_size = get<0>(centered_coords).size();
  result->initialize(grid_size);
  const double xp = particle_position[0];
  const double yp = particle_position[1];
  const double zp = particle_position[2];
  const double xpdot = particle_velocity[0];
  const double ypdot = particle_velocity[1];
  const double zpdot = particle_velocity[2];

  const double xpddot = particle_acceleration[0];
  const double ypddot = particle_acceleration[1];
  const double zpddot = particle_acceleration[2];

  const double rp = get(magnitude(particle_position));
  const double rpdot = (xp * xpdot + yp * ypdot + zp * zpdot) / rp;

  const auto& Dx = get<0>(centered_coords);
  const auto& Dy = get<1>(centered_coords);
  const auto& Dz = get<2>(centered_coords);

  const double M = bh_mass;
  const double a = bh_spin[2];

  DynamicBuffer<DataVector> temps(91, grid_size);

  const double d_0 = a * a;
  const double d_1 = rp * rp;
  const double d_2 = d_0 + d_1;
  const double d_3 = 1.0 / d_2;
  const double d_4 = a * yp;
  const double d_5 = rp * xp;
  const double d_6 = d_4 + d_5;
  const double d_7 = d_3 * d_6;
  const double d_8 = d_7 * xpdot;
  const double d_9 = a * xp - rp * yp;
  const double d_10 = -d_9;
  const double d_11 = d_10 * d_3;
  const double d_12 = d_11 * ypdot;
  const double d_13 = 1.0 / rp;
  const double d_14 = zp * zpdot;
  const double d_15 = d_13 * d_14;
  const double d_16 = d_15 + 1;
  const double d_17 = d_12 + d_16;
  const double d_18 = d_17 + d_8;
  const double d_19 = d_18 * d_18;
  const double d_20 = rp * rp * rp;
  const double d_21 = rp * rp * rp * rp;
  const double d_22 = zp * zp;
  const double d_23 = d_0 * d_22;
  const double d_24 = d_21 + d_23;
  const double d_25 = 1.0 / d_24;
  const double d_26 = 2 * d_25;
  const double d_27 = d_20 * d_26;
  const double d_28 = -1 + xpdot * xpdot + ypdot * ypdot + zpdot * zpdot;
  const double d_29 = M * d_19 * d_27 + d_28;
  const double d_30 = 1.0 / d_29;
  const double d_31 = d_3 * d_9;
  const double d_32 = 2 * d_31;
  const double d_33 = d_13 * zp;
  const double d_34 = M * d_25;
  const double d_35 = d_20 * d_34;
  const double d_36 = 2 * d_7;
  const double d_37 = d_35 * d_36;
  const double d_38 = d_9 * d_9;
  const double d_39 = 1.0 / (d_2 * d_2);
  const double d_40 = M * d_27;
  const double d_41 = d_39 * d_40;
  const double d_42 = M * d_26;
  const double d_43 = d_1 * d_42;
  const double d_44 = 2 * rp;
  const double d_45 = d_22 * d_44;
  const double d_46 = d_34 * d_45;
  const double d_47 = d_6 * d_6;
  const double d_48 = d_41 * d_47;
  const double d_49 = 4 * d_35;
  const double d_50 = 4 * d_1;
  const double d_51 = 3 * d_23;
  const double d_52 = d_21 - d_51;
  const double d_53 = xp * xp;
  const double d_54 = yp * yp;
  const double d_55 = d_22 + d_53 + d_54;
  const double d_56 = -d_0 + d_55;
  const double d_57 = 4 * d_23 + d_56 * d_56;
  const double d_58 = sqrt(d_57);
  const double d_59 = d_56 + d_58;
  const double d_60 = sqrt(d_59);
  const double d_61 = 1.0 / d_58;
  const double d_62 = M_SQRT2;
  const double d_63 = d_61 * d_62;
  const double d_64 = d_60 * d_63;
  const double d_65 = d_64 * xp;
  const double d_66 = d_52 * d_65;
  const double d_67 = d_52 * d_64;
  const double d_68 = d_67 * yp;
  const double d_69 = 4 * rp;
  const double d_70 = d_0 * d_69;
  const double d_71 = d_0 + d_55 + d_58;
  const double d_72 = 1.0 / d_60;
  const double d_73 = d_71 * d_72;
  const double d_74 = d_63 * d_73;
  const double d_75 = d_52 * d_74 + d_70;
  const double d_76 = 1.0 / d_1;
  const double d_77 = d_60 * d_76;
  const double d_78 = d_77 * xp;
  const double d_79 = 2 * d_64;
  const double d_80 = d_5 * d_79;
  const double d_81 = d_44 + d_53 * d_64;
  const double d_82 = -d_2 * d_81 + d_6 * d_80;
  const double d_83 = d_80 * d_9;
  const double d_84 = 2 * a;
  const double d_85 = -d_60 * d_61 * d_62 * xp * yp + d_84;
  const double d_86 = d_77 * yp;
  const double d_87 = rp * yp;
  const double d_88 = d_79 * d_87;
  const double d_89 = d_65 * yp + d_84;
  const double d_90 = -d_2 * d_89 + d_6 * d_88;
  const double d_91 = d_44 + d_54 * d_64;
  const double d_92 = d_10 * d_88 - d_2 * d_91;
  const double d_93 = 2 * d_33;
  const double d_94 = -d_13 * d_22 * d_74 + 2;
  const double d_95 = d_4 * d_44;
  const double d_96 = d_0 - d_1;
  const double d_97 = d_95 - d_96 * xp;
  const double d_98 = d_71 * zp;
  const double d_99 = d_63 * d_72;
  const double d_100 = d_98 * d_99;
  const double d_101 = d_0 * yp - d_1 * yp + d_5 * d_84;
  const double d_102 = -d_31 * ypdot;
  const double d_103 = d_16 + d_8;
  const double d_104 = d_102 + d_103;
  const double d_105 = xp * xpdot;
  const double d_106 = yp * ypdot;
  const double d_107 = d_105 * d_67 + d_106 * d_67 + d_14 * d_75;
  const double d_108 = d_107 * d_25;
  const double d_109 = d_63 * zp;
  const double d_110 = d_109 * d_86;
  const double d_111 = d_109 * d_78;
  const double d_112 = -d_97;
  const double d_113 = d_100 * d_39;
  const double d_114 = d_112 * d_113;
  const double d_115 = d_101 * d_113;
  const double d_116 = -d_90;
  const double d_117 = d_116 * d_39;
  const double d_118 = -d_85;
  const double d_119 = d_118 * d_2 + d_83;
  const double d_120 = d_119 * d_39;
  const double d_121 = -d_82;
  const double d_122 = -d_92;
  const double d_123 = 2 * d_11;
  const double d_124 = d_1 * d_34;
  const double d_125 = -d_29;
  const double d_126 = 1.0 / d_125;
  const double d_127 = d_10 * d_10;
  const double d_128 = rp * zpdot;
  const double d_129 = rpdot * zp;
  const double d_130 = d_128 - d_129;
  const double d_131 = 4 * d_130 * zp;
  const double d_132 = 1.0 / (d_24 * d_24);
  const double d_133 = d_21 * rpdot;
  const double d_134 = d_0 * d_14;
  const double d_135 = d_133 + d_134 * d_44 - d_51 * rpdot;
  const double d_136 = M * d_135;
  const double d_137 = d_132 * d_136;
  const double d_138 = 2 * d_137 * d_22;
  const double d_139 = d_1 * d_132;
  const double d_140 = 2 * d_136;
  const double d_141 = d_139 * d_140;
  const double d_142 = d_141 * d_39 * d_47;
  const double d_143 = d_44 * rpdot;
  const double d_144 = rp * ypdot;
  const double d_145 = rpdot * yp;
  const double d_146 = a * xpdot;
  const double d_147 = d_144 + d_145 - d_146;
  const double d_148 = -d_10 * d_143 + d_147 * d_2;
  const double d_149 = d_127 * d_39;
  const double d_150 = d_141 * d_149;
  const double d_151 = rp * xpdot;
  const double d_152 = rpdot * xp;
  const double d_153 = a * ypdot + d_152;
  const double d_154 = d_151 + d_153;
  const double d_155 = -d_143 * d_6 + d_154 * d_2;
  const double d_156 = 1.0 / (d_2 * d_2 * d_2);
  const double d_157 = d_156 * d_49;
  const double d_158 = d_155 * d_157 * d_6;
  const double d_159 = d_10 * d_148 * d_157;
  const double d_160 = d_130 * d_76;
  const double d_161 = d_8 + 1;
  const double d_162 = d_43 * zp;
  const double d_163 = M * d_25 * d_45 * zpdot + zpdot;
  const double d_164 = d_124 * zp;
  const double d_165 = 2 * zpdot;
  const double d_166 = 2 * xpdot;
  const double d_167 = d_166 * d_39;
  const double d_168 = d_167 * d_35 * d_47 + xpdot;
  const double d_169 = d_123 * d_124;
  const double d_170 = 2 * ypdot;
  const double d_171 = d_170 * d_35;
  const double d_172 = d_123 * d_35;
  const double d_173 = d_148 * d_39;
  const double d_174 = d_136 * d_139;
  const double d_175 = d_14 * d_74;
  const double d_176 = d_116 * ypdot;
  const double d_177 = d_39 * xpdot;
  const double d_178 = d_119 * xpdot;
  const double d_179 = d_39 * ypdot;
  const double d_180 = d_33 * d_64;
  const double d_181 = d_123 * ypddot + d_36 * xpddot + d_93 * zpddot;
  const double d_182 = d_18 * d_40;
  const double d_183 =
      M * d_1 * d_107 * d_132 * d_19 - d_165 * zpddot - d_166 * xpddot -
      d_170 * ypddot -
      d_182 * (d_13 * zpdot * (-d_105 * d_180 - d_106 * d_180 + d_94 * zpdot) +
               d_177 * (d_112 * d_175 + d_121 * xpdot + d_176) +
               d_179 * (d_101 * d_175 + d_122 * ypdot + d_178) + d_181);
  const double d_184 = rp * zp;
  const double d_185 = d_184 * d_75;
  const double d_186 = d_185 * d_25;
  const double d_187 = d_39 * d_73;
  const double d_188 = d_101 * d_187;
  const double d_189 = d_188 + d_86;
  const double d_190 = d_1 * d_18;
  const double d_191 = d_112 * d_187;
  const double d_192 = d_14 * d_63;
  const double d_193 = d_44 * xpdot;
  const double d_194 = d_44 * ypdot;
  const double d_195 = sqrt(d_125);
  const double d_196 = d_11 * ypddot + d_148 * d_179 + d_155 * d_177 +
                       d_160 * zpdot + d_33 * zpddot + d_7 * xpddot;
  const double d_197 = 2 * xp;
  const double d_198 = rp * rp * rp * rp * rp * rp * rp * rp;
  const double d_199 = 3 * (a * a * a * a);
  const double d_200 = d_199 * (zp * zp * zp * zp);
  const double d_201 = rp * rp * rp * rp * rp;
  const double d_202 = zp * zp * zp;
  const double d_203 = -d_128 * d_199 * d_202 - 12 * d_133 * d_23 +
                       5 * d_134 * d_201 + d_198 * rpdot + d_200 * rpdot;
  const double d_204 = d_203 * d_60;
  const double d_205 = d_57 * d_59;
  const double d_206 = d_205 * xpdot;
  const double d_207 = d_105 + d_106 + d_14;
  const double d_208 = 2 * d_134 + d_207 * d_56;
  const double d_209 = d_208 * d_59;
  const double d_210 = d_197 * d_209;
  const double d_211 = d_207 + d_208 * d_61;
  const double d_212 = d_211 * d_57;
  const double d_213 = d_212 * xp;
  const double d_214 = rp * (-d_198 + d_200 + 2 * d_21 * d_23);
  const double d_215 = d_214 * d_72 * 1.0 / d_57;
  const double d_216 = d_197 * d_204 + d_215 * (d_206 - d_210 + d_213);
  const double d_217 = 2 * yp;
  const double d_218 = d_205 * ypdot;
  const double d_219 =
      d_204 * d_217 + d_215 * (-d_209 * d_217 + d_212 * yp + d_218);
  const double d_220 = 2 * d_98;
  const double d_221 = d_205 * zpdot;
  const double d_222 = 2 * d_209 * d_98;
  const double d_223 = d_212 * d_98;
  const double d_224 = d_212 * zp;
  const double d_225 = pow(d_59, -3.0 / 2.0);
  const double d_226 = pow(d_57, -3.0 / 2.0);
  const double d_227 = d_226 * d_62;
  const double d_228 = d_225 * d_227;
  const double d_229 =
      d_203 * d_220 * d_99 +
      d_214 * d_228 * (d_221 * d_71 - d_222 - d_223 + 2 * d_224 * d_59) +
      d_70 * (-3 * d_0 * d_202 * rpdot + d_128 * d_51 + 5 * d_129 * d_21 -
              d_201 * zpdot);
  const double d_230 = 2 * d_129;
  const double d_231 = d_129 * d_69;
  const double d_232 = d_231 * d_74;
  const double d_233 = d_146 - 2 * rpdot * yp;
  const double d_234 = d_1 * xpdot + d_153 * d_44;
  const double d_235 = 2 * rpdot;
  const double d_236 = d_0 * ypdot - d_1 * ypdot + d_151 * d_84 + d_235 * d_9;
  const double d_237 = d_211 * d_63;
  const double d_238 = d_237 * d_72;
  const double d_239 = 2 * d_238;
  const double d_240 = d_239 * zp;
  const double d_241 = d_227 * d_72;
  const double d_242 = d_208 * d_220 * d_241;
  const double d_243 = d_225 * d_237 * d_98;
  const double d_244 = 1.0 / d_20;
  const double d_245 = 2 * d_3;
  const double d_246 = d_109 * d_77;
  const double d_247 = d_69 * rpdot;
  const double d_248 = d_119 * d_247;
  const double d_249 = d_193 * d_64;
  const double d_250 = d_152 * d_79;
  const double d_251 = d_208 * d_227;
  const double d_252 = d_251 * d_60;
  const double d_253 = 2 * d_252;
  const double d_254 = -d_147;
  const double d_255 = 4 * d_252;
  const double d_256 = d_5 * d_9;
  const double d_257 = d_206 * yp - d_210 * yp + d_213 * yp + d_218 * xp;
  const double d_258 = d_118 * d_143 + d_2 * d_241 * d_257 + d_239 * d_256 +
                       d_249 * d_9 + d_250 * d_9 + d_254 * d_80 - d_255 * d_256;
  const double d_259 = d_116 * d_247;
  const double d_260 = 2 * d_145;
  const double d_261 = d_10 * d_87;
  const double d_262 = d_6 * d_64;
  const double d_263 =
      -d_154 * d_88 - d_194 * d_262 + d_2 * d_226 * d_257 * d_62 * d_72 +
      4 * d_208 * d_226 * d_6 * d_60 * d_62 * rp * yp - d_239 * d_6 * d_87 -
      d_260 * d_262 + 2 * d_89 * rp * rpdot;
  const double d_264 = 4 * d_18;
  const double d_265 = d_2 * d_258 - d_248;
  const double d_266 = d_2 * d_263 - d_259;
  const double d_267 = d_205 * zp;
  const double d_268 = d_241 * d_244;
  const double d_269 =
      d_268 * (-d_151 * d_267 + 2 * d_208 * d_59 * rp * xp * zp - d_221 * d_5 -
               d_224 * d_5 + 2 * d_57 * d_59 * rpdot * xp * zp);
  const double d_270 =
      d_268 * (-d_144 * d_267 + 2 * d_208 * d_59 * rp * yp * zp - d_221 * d_87 -
               d_224 * d_87 + 2 * d_57 * d_59 * rpdot * yp * zp);
  const double d_271 = d_205 * d_231 * d_71;
  const double d_272 = d_112 * d_2;
  const double d_273 = d_156 * d_228;
  const double d_274 =
      d_273 * (2 * d_112 * d_2 * d_211 * d_57 * d_59 * zp +
               d_112 * d_2 * d_57 * d_59 * d_71 * zpdot - d_112 * d_271 +
               d_2 * d_57 * d_59 * d_71 * zp * (a * d_233 - d_234) -
               d_222 * d_272 - d_223 * d_272);
  const double d_275 = d_101 * d_2;
  const double d_276 =
      d_273 *
      (2 * d_101 * d_2 * d_211 * d_57 * d_59 * zp +
       d_101 * d_2 * d_57 * d_59 * d_71 * zpdot - d_101 * d_271 +
       d_2 * d_236 * d_57 * d_59 * d_71 * zp - d_222 * d_275 - d_223 * d_275);
  const double d_277 = d_34 * rp;
  DataVector& dv_0 = temps.at(0);
  dv_0 = Dy * zpdot;
  DataVector& dv_1 = temps.at(1);
  dv_1 = Dz * ypdot;
  DataVector& dv_2 = temps.at(2);
  dv_2 = d_35 * (Dy + d_33 * dv_0 + d_33 * dv_1);
  DataVector& dv_3 = temps.at(3);
  dv_3 = Dx * ypdot;
  DataVector& dv_4 = temps.at(4);
  dv_4 = Dy * xpdot;
  DataVector& dv_5 = temps.at(5);
  dv_5 = dv_3 + dv_4;
  DataVector& dv_6 = temps.at(6);
  dv_6 = Dx * zpdot;
  DataVector& dv_7 = temps.at(7);
  dv_7 = Dz * xpdot;
  DataVector& dv_8 = temps.at(8);
  dv_8 = d_33 * dv_6 + d_33 * dv_7;
  DataVector& dv_9 = temps.at(9);
  dv_9 = Dy * ypdot;
  DataVector& dv_10 = temps.at(10);
  dv_10 = d_41 * dv_9;
  DataVector& dv_11 = temps.at(11);
  dv_11 = Dz * zp;
  DataVector& dv_12 = temps.at(12);
  dv_12 = Dz * zpdot;
  DataVector& dv_13 = temps.at(13);
  dv_13 = Dx * xpdot;
  DataVector& dv_14 = temps.at(14);
  dv_14 = dv_12 + dv_13 + dv_9;
  DataVector& dv_15 = temps.at(15);
  dv_15 = d_43 * dv_11 + d_46 * dv_12 + d_48 * dv_13 + dv_14;
  DataVector& dv_16 = temps.at(16);
  dv_16 =
      -d_32 * dv_2 + d_37 * (Dx - d_31 * dv_5 + dv_8) + d_38 * dv_10 + dv_15;
  DataVector& dv_17 = temps.at(17);
  dv_17 = dv_16 * dv_16;
  DataVector& dv_18 = temps.at(18);
  dv_18 = Dy * d_31;
  DataVector& dv_19 = temps.at(19);
  dv_19 = -Dz * d_13 * zp + dv_18;
  DataVector& dv_20 = temps.at(20);
  dv_20 = -dv_19;
  DataVector& dv_21 = temps.at(21);
  dv_21 = Dx * d_7;
  DataVector& dv_22 = temps.at(22);
  dv_22 = d_49 * dv_21;
  DataVector& dv_23 = temps.at(23);
  dv_23 = Dy * Dy;
  DataVector& dv_24 = temps.at(24);
  dv_24 = d_41 * dv_23;
  DataVector& dv_25 = temps.at(25);
  dv_25 = d_34 * d_50 * dv_11;
  DataVector& dv_26 = temps.at(26);
  dv_26 = Dx * Dx;
  DataVector& dv_27 = temps.at(27);
  dv_27 = Dz * Dz;
  DataVector& dv_28 = temps.at(28);
  dv_28 = d_34 * dv_27;
  DataVector& dv_29 = temps.at(29);
  dv_29 = d_45 * dv_28 + d_48 * dv_26 + dv_23 + dv_26 + dv_27;
  DataVector& dv_30 = temps.at(30);
  dv_30 = d_38 * dv_24 - dv_18 * dv_25 + dv_29;
  DataVector& dv_31 = temps.at(31);
  dv_31 = -d_30 * dv_17 + dv_20 * dv_22 + dv_30;
  DataVector& dv_32 = temps.at(32);
  dv_32 = sqrt(dv_31);
  DataVector& dv_33 = temps.at(33);
  dv_33 = 1.0 / dv_32;
  DataVector& dv_34 = temps.at(34);
  dv_34 = d_13 * dv_11;
  DataVector& dv_35 = temps.at(35);
  dv_35 = -dv_18 + dv_21 + dv_34;
  DataVector& dv_36 = temps.at(36);
  dv_36 = Dx * d_66;
  DataVector& dv_37 = temps.at(37);
  dv_37 = Dy * d_68;
  DataVector& dv_38 = temps.at(38);
  dv_38 = d_75 * dv_11;
  DataVector& dv_39 = temps.at(39);
  dv_39 = dv_36 + dv_37 + dv_38;
  DataVector& dv_40 = temps.at(40);
  dv_40 = d_25 * dv_39;
  DataVector& dv_41 = temps.at(41);
  dv_41 = d_63 * dv_11;
  DataVector& dv_42 = temps.at(42);
  dv_42 = d_78 * dv_41;
  DataVector& dv_43 = temps.at(43);
  dv_43 = Dx * d_39;
  DataVector& dv_44 = temps.at(44);
  dv_44 = d_86 * dv_41;
  DataVector& dv_45 = temps.at(45);
  dv_45 = Dy * d_39;
  DataVector& dv_46 = temps.at(46);
  dv_46 = d_100 * dv_43;
  DataVector& dv_47 = temps.at(18);
  dv_47 = 2 * dv_18;
  DataVector& dv_48 = temps.at(47);
  dv_48 = 2 * dv_21;
  DataVector& dv_49 = temps.at(48);
  dv_49 = 2 * dv_34 + dv_48;
  DataVector& dv_50 = temps.at(49);
  dv_50 = Dy * d_11;
  DataVector& dv_51 = temps.at(34);
  dv_51 = dv_34 + dv_50;
  DataVector& dv_52 = temps.at(50);
  dv_52 = dv_21 + dv_51;
  DataVector& dv_53 = temps.at(51);
  dv_53 = d_108 * dv_52;
  DataVector& dv_54 = temps.at(52);
  dv_54 = -d_110 * dv_0 + d_110 * dv_1 - d_111 * (dv_6 - dv_7) - d_114 * dv_6 +
          d_114 * dv_7 - d_115 * dv_0 + d_115 * dv_1 - d_117 * dv_3 +
          d_117 * dv_4 + 2 * d_12 + d_120 * (dv_3 - dv_4) + 2 * d_15 + 2 * d_8 +
          2;
  DataVector& dv_55 = temps.at(53);
  dv_55 = d_44 * dv_54;
  DataVector& dv_56 = temps.at(54);
  dv_56 = Dz * d_13 * d_94;
  DataVector& dv_57 = temps.at(55);
  dv_57 = d_112 * dv_43;
  DataVector& dv_58 = temps.at(56);
  dv_58 = d_100 * dv_45;
  DataVector& dv_59 = temps.at(57);
  dv_59 = d_100 * dv_57 + d_101 * dv_58;
  DataVector& dv_60 = temps.at(58);
  dv_60 = -d_93 + dv_59;
  DataVector& dv_61 = temps.at(59);
  dv_61 = d_121 * dv_43;
  DataVector& dv_62 = temps.at(60);
  dv_62 = Dy * d_120 - dv_42;
  DataVector& dv_63 = temps.at(61);
  dv_63 = -d_36 + dv_62;
  DataVector& dv_64 = temps.at(62);
  dv_64 = d_122 * dv_45;
  DataVector& dv_65 = temps.at(63);
  dv_65 = d_116 * dv_43 - dv_44;
  DataVector& dv_66 = temps.at(64);
  dv_66 = -d_123 + dv_65;
  DataVector& dv_67 = temps.at(65);
  dv_67 = Dx * (dv_61 + dv_63) + Dy * (dv_64 + dv_66) + Dz * (dv_56 + dv_60);
  DataVector& dv_68 = temps.at(66);
  dv_68 = d_44 * dv_67;
  DataVector& dv_69 = temps.at(67);
  dv_69 = 2 * dv_52;
  DataVector& dv_70 = temps.at(68);
  dv_70 = d_18 * dv_40;
  DataVector& dv_71 = temps.at(69);
  dv_71 = d_18 * dv_68 - dv_69 * dv_70;
  DataVector& dv_72 = temps.at(70);
  dv_72 = dv_52 * (dv_53 + dv_55) + dv_71;
  DataVector& dv_73 = temps.at(35);
  dv_73 = -d_30 * dv_72 * (d_104 * d_40 * dv_35 + dv_14) +
          2 * dv_35 * rp *
              (Dx * (Dy * d_39 * (-d_2 * d_85 + d_83) - d_36 - d_82 * dv_43 -
                     dv_42) +
               Dy * (2 * d_3 * d_9 - d_90 * dv_43 - d_92 * dv_45 - dv_44) +
               Dz * (Dy * d_101 * d_39 * d_61 * d_62 * d_71 * d_72 * zp +
                     Dz * d_13 * d_94 - d_93 - d_97 * dv_46) -
               dv_47 + dv_49) -
          dv_40 * dv_35 * dv_35;
  DataVector& dv_74 = temps.at(30);
  dv_74 = pow(
      -dv_17 * 1.0 / (d_28 + d_40 * (d_104 * d_104)) - dv_19 * dv_22 + dv_30,
      -3.0 / 2.0);
  DataVector& dv_75 = temps.at(5);
  dv_75 = Dx + d_11 * dv_5;
  DataVector& dv_76 = temps.at(10);
  dv_76 = d_123 * dv_2 + d_127 * dv_10 + d_37 * (dv_75 + dv_8) + dv_15;
  DataVector& dv_77 = temps.at(15);
  dv_77 = dv_76 * dv_76;
  DataVector& dv_78 = temps.at(24);
  dv_78 = d_126 * dv_77 + d_127 * dv_24 + dv_22 * dv_51 + dv_25 * dv_50 + dv_29;
  DataVector& dv_79 = temps.at(29);
  dv_79 = sqrt(dv_78);
  DataVector& dv_80 = temps.at(2);
  dv_80 = 1.0 / dv_79;
  DataVector& dv_81 = temps.at(28);
  dv_81 = d_131 * dv_28;
  DataVector& dv_82 = temps.at(8);
  dv_82 = Dz * d_34;
  DataVector& dv_83 = temps.at(19);
  dv_83 = d_130 * dv_82;
  DataVector& dv_84 = temps.at(17);
  dv_84 = d_69 * dv_50;
  DataVector& dv_85 = temps.at(42);
  dv_85 = dv_83 * dv_84;
  DataVector& dv_86 = temps.at(27);
  dv_86 = d_138 * dv_27;
  DataVector& dv_87 = temps.at(44);
  dv_87 = d_137 * dv_11;
  DataVector& dv_88 = temps.at(17);
  dv_88 = dv_84 * dv_87;
  DataVector& dv_89 = temps.at(71);
  dv_89 = d_142 * dv_26;
  DataVector& dv_90 = temps.at(72);
  dv_90 = d_148 * dv_45;
  DataVector& dv_91 = temps.at(25);
  dv_91 = dv_25 * dv_90;
  DataVector& dv_92 = temps.at(73);
  dv_92 = d_150 * dv_23;
  DataVector& dv_93 = temps.at(26);
  dv_93 = d_158 * dv_26;
  DataVector& dv_94 = temps.at(23);
  dv_94 = d_159 * dv_23;
  DataVector& dv_95 = temps.at(74);
  dv_95 = d_155 * dv_43;
  DataVector& dv_96 = temps.at(75);
  dv_96 = d_49 * dv_51 * dv_95;
  DataVector& dv_97 = temps.at(21);
  dv_97 = d_137 * d_50 * dv_21 * dv_51;
  DataVector& dv_98 = temps.at(72);
  dv_98 = Dz * d_160 + dv_90;
  DataVector& dv_99 = temps.at(22);
  dv_99 = dv_22 * dv_98;
  DataVector& dv_100 = temps.at(10);
  dv_100 = d_126 * dv_76;
  DataVector& dv_101 = temps.at(49);
  dv_101 = 2 * dv_50;
  DataVector& dv_102 = temps.at(47);
  dv_102 = Dz + d_164 * dv_48 + d_45 * dv_82;
  DataVector& dv_103 = temps.at(8);
  dv_103 = d_164 * dv_101 + dv_100 * (d_162 * (d_12 + d_161) + d_163) + dv_102;
  DataVector& dv_104 = temps.at(76);
  dv_104 = d_40 * dv_43;
  DataVector& dv_105 = temps.at(77);
  dv_105 = Dx + d_47 * dv_104;
  DataVector& dv_106 = temps.at(34);
  dv_106 = d_37 * dv_51 + dv_100 * (d_168 + d_17 * d_37) + dv_105;
  DataVector& dv_107 = temps.at(76);
  dv_107 = d_6 * dv_104;
  DataVector& dv_108 = temps.at(78);
  dv_108 = Dy + d_10 * dv_107 + d_127 * d_40 * dv_45 + d_169 * dv_11 +
           dv_100 * (d_103 * d_172 + d_149 * d_171 + ypdot);
  DataVector& dv_109 = temps.at(79);
  dv_109 = Dz * zpddot;
  DataVector& dv_110 = temps.at(80);
  dv_110 = Dy * zpddot;
  DataVector& dv_111 = temps.at(81);
  dv_111 = Dz * ypddot;
  DataVector& dv_112 = temps.at(82);
  dv_112 = dv_0 + dv_1;
  DataVector& dv_113 = temps.at(83);
  dv_113 = Dx * zpddot;
  DataVector& dv_114 = temps.at(84);
  dv_114 = Dz * xpddot;
  DataVector& dv_115 = temps.at(85);
  dv_115 = Dx * ypddot;
  DataVector& dv_116 = temps.at(86);
  dv_116 = Dy * xpddot;
  DataVector& dv_117 = temps.at(87);
  dv_117 = Dy + d_33 * dv_112;
  DataVector& dv_118 = temps.at(5);
  dv_118 = d_33 * (dv_6 + dv_7) + dv_75;
  DataVector& dv_119 = temps.at(88);
  dv_119 = Dx * xpddot;
  DataVector& dv_120 = temps.at(89);
  dv_120 = Dy * ypddot;
  DataVector& dv_121 = temps.at(90);
  dv_121 = dv_109 + dv_119 + dv_120;
  DataVector& dv_122 = temps.at(89);
  dv_122 =
      2 * dv_100 *
      (-d_123 * d_174 * dv_117 + d_127 * d_41 * dv_120 + d_131 * d_34 * dv_12 -
       d_138 * dv_12 - d_142 * dv_13 - d_150 * dv_9 + d_155 * d_41 * dv_118 +
       d_158 * dv_13 + d_159 * dv_9 +
       d_169 * (d_13 * d_130 * dv_112 + zp * (dv_110 + dv_111)) +
       d_173 * d_40 * dv_117 - d_174 * d_36 * dv_118 +
       d_37 * (d_11 * (dv_115 + dv_116) + d_160 * dv_6 + d_160 * dv_7 +
               d_173 * dv_3 + d_173 * dv_4 + d_33 * (dv_113 + dv_114)) +
       d_44 * dv_83 - d_44 * dv_87 + d_46 * dv_109 + d_48 * dv_119 + dv_121);
  DataVector& dv_123 = temps.at(15);
  dv_123 = d_183 * dv_77 * 1.0 / (d_125 * d_125);
  DataVector& dv_124 = temps.at(10);
  dv_124 = dv_80 * (d_165 * dv_103 + d_166 * dv_106 + d_170 * dv_108 - dv_122 +
                    dv_123 - dv_81 - dv_85 + dv_86 + dv_88 + dv_89 - dv_91 +
                    dv_92 - dv_93 - dv_94 - dv_96 + dv_97 - dv_99);
  DataVector& dv_125 = temps.at(36);
  dv_125 = dv_52 * (-d_25 * dv_36 - d_25 * dv_37 - d_25 * dv_38 + 4 * rp);
  DataVector& dv_126 = temps.at(70);
  dv_126 = -dv_72;
  DataVector& dv_127 = temps.at(38);
  dv_127 = d_35 * dv_69;
  DataVector& dv_128 = temps.at(14);
  dv_128 = d_18 * dv_127 + dv_14;
  DataVector& dv_129 = temps.at(37);
  dv_129 = d_126 * dv_128;
  DataVector& dv_130 = temps.at(5);
  dv_130 = dv_126 * dv_129;
  DataVector& dv_131 = temps.at(87);
  dv_131 = dv_52 * dv_52;
  DataVector& dv_132 = temps.at(48);
  dv_132 = dv_101 + dv_49 + dv_67;
  DataVector& dv_133 = temps.at(49);
  dv_133 = dv_80 * (-dv_130 - dv_131 * dv_40 + 2 * dv_132 * dv_52 * rp);
  DataVector& dv_134 = temps.at(54);
  dv_134 = -Dx * d_111 - Dy * d_110 + 2 * dv_56;
  DataVector& dv_135 = temps.at(48);
  dv_135 = d_44 * dv_132;
  DataVector& dv_136 = temps.at(13);
  dv_136 = d_18 * dv_52;
  DataVector& dv_137 = temps.at(9);
  dv_137 = 2 * dv_129;
  DataVector& dv_138 = temps.at(19);
  dv_138 = d_126 * dv_126;
  DataVector& dv_139 = temps.at(8);
  dv_139 =
      dv_103 * dv_133 * rp -
      dv_79 *
          (d_1 * dv_69 * (dv_134 + dv_59) - d_186 * dv_131 + dv_135 * zp +
           dv_137 * (d_1 * d_109 * dv_52 *
                         (d_105 * d_77 + d_112 * d_177 * d_73 + d_189 * ypdot) +
                     d_184 * dv_54 - d_186 * dv_136 + d_190 * (dv_134 + dv_60) +
                     dv_53 * zp - dv_70 * zp) -
           dv_138 * rp * (d_190 * d_42 * zp + zpdot) - dv_40 * dv_69 * zp);
  DataVector& dv_140 = temps.at(58);
  dv_140 = d_25 * dv_131;
  DataVector& dv_141 = temps.at(40);
  dv_141 = dv_40 * dv_52;
  DataVector& dv_142 = temps.at(59);
  dv_142 = d_116 * dv_45 + d_191 * dv_41 + 2 * dv_61;
  DataVector& dv_143 = temps.at(57);
  dv_143 = d_44 * dv_52;
  DataVector& dv_144 = temps.at(13);
  dv_144 = d_25 * dv_136;
  DataVector& dv_145 = temps.at(54);
  dv_145 = dv_52 * rp;
  DataVector& dv_146 = temps.at(61);
  dv_146 = dv_106 * dv_133 -
           dv_79 * (-d_36 * dv_141 - d_66 * dv_140 + d_7 * dv_135 +
                    dv_137 * (d_18 * rp * (dv_142 + dv_63) - d_66 * dv_144 +
                              d_7 * dv_53 + d_7 * dv_54 * rp - d_7 * dv_70 +
                              dv_145 * (d_119 * d_39 * ypdot - d_176 * d_39 -
                                        d_192 * (d_191 + d_78))) -
                    dv_138 * (d_18 * d_37 + xpdot) + dv_143 * (dv_142 + dv_62));
  DataVector& dv_147 = temps.at(41);
  dv_147 = Dx * d_120 + d_188 * dv_41 + 2 * dv_64;
  DataVector& dv_148 = temps.at(78);
  dv_148 =
      dv_108 * dv_133 -
      dv_79 *
          (d_11 * dv_135 - d_123 * dv_141 - d_68 * dv_140 +
           dv_137 *
               (d_10 * d_107 * d_25 * d_3 * dv_52 + d_10 * d_3 * dv_54 * rp -
                d_11 * dv_70 + d_18 * rp * (dv_147 + dv_66) - d_68 * dv_144 -
                dv_145 * (-d_116 * d_177 + d_178 * d_39 + d_189 * d_192)) -
           dv_138 * (d_172 * d_18 + ypdot) + dv_143 * (dv_147 + dv_65));
  DataVector& dv_149 = temps.at(74);
  dv_149 = dv_95 + dv_98;
  DataVector& dv_150 = temps.at(72);
  dv_150 = d_69 * dv_149;
  DataVector& dv_151 = temps.at(58);
  dv_151 = d_132 * dv_52;
  DataVector& dv_152 = temps.at(13);
  dv_152 = dv_151 * (Dx * d_216 * d_63 + Dy * d_219 * d_63 + Dz * d_229);
  DataVector& dv_153 = temps.at(64);
  dv_153 = Dx * d_156;
  DataVector& dv_154 = temps.at(41);
  dv_154 = Dy * d_156;
  DataVector& dv_155 = temps.at(9);
  dv_155 = d_101 * dv_45;
  DataVector& dv_156 = temps.at(63);
  dv_156 = Dz * d_244;
  DataVector& dv_157 = temps.at(12);
  dv_157 = d_63 * dv_12;
  DataVector& dv_158 = temps.at(19);
  dv_158 = d_238 * d_76 * dv_11;
  DataVector& dv_159 = temps.at(63);
  dv_159 =
      d_1 *
      (Dx * (Dx * d_39 *
                 (-d_154 * d_80 +
                  d_2 * (d_105 * d_79 + d_235 + d_238 * d_53 - d_253 * d_53) +
                  4 * d_208 * d_226 * d_6 * d_60 * d_62 * rp * xp -
                  d_239 * d_5 * d_6 - d_249 * d_6 - d_250 * d_6 +
                  2 * d_81 * rp * rpdot) +
             Dy * d_258 * d_39 +
             2 * Dz * d_208 * d_226 * d_60 * d_62 * d_76 * xp * zp +
             2 * Dz * d_244 * d_60 * d_61 * d_62 * rpdot * xp * zp -
             d_121 * d_247 * dv_153 - d_154 * d_245 - d_246 * dv_7 -
             d_248 * dv_154 + 4 * d_39 * d_6 * rp * rpdot - d_78 * dv_157 -
             dv_158 * xp) +
       Dy * (d_10 * d_247 * d_39 - d_122 * d_247 * dv_154 +
             d_230 * d_64 * dv_156 * yp + d_245 * d_254 - d_246 * dv_1 +
             2 * d_251 * d_86 * dv_11 - d_259 * dv_153 + d_263 * dv_43 -
             d_86 * dv_157 - dv_158 * yp +
             dv_45 *
                 (-d_10 * d_260 * d_64 + d_143 * d_91 - d_147 * d_88 +
                  d_194 * d_64 * d_9 +
                  d_2 * (d_106 * d_79 + d_235 + d_238 * d_54 - d_253 * d_54) -
                  d_239 * d_261 + d_255 * d_261)) +
       Dz * (-d_101 * d_232 * dv_154 - d_13 * d_165 + d_188 * d_63 * dv_0 +
             d_191 * d_63 * dv_6 + d_230 * d_76 +
             d_232 * dv_153 * (d_95 - d_96 * xp) + d_236 * dv_58 +
             d_240 * dv_155 + d_240 * dv_57 - d_242 * dv_155 - d_242 * dv_57 -
             d_243 * dv_155 - d_243 * dv_57 +
             dv_156 *
                 (-d_143 - d_175 * d_44 +
                  2 * d_208 * d_22 * d_226 * d_62 * d_71 * d_72 * rp +
                  d_211 * d_22 * d_225 * d_61 * d_62 * d_71 * rp +
                  2 * d_22 * d_61 * d_62 * d_71 * d_72 * rpdot - d_238 * d_45) -
             dv_46 * (-a * d_233 + d_234)));
  DataVector& dv_160 = temps.at(3);
  dv_160 = d_156 * dv_3;
  DataVector& dv_161 = temps.at(4);
  dv_161 = d_156 * dv_4;
  DataVector& dv_162 = temps.at(16);
  dv_162 = d_30 * dv_16;
  DataVector& dv_163 = temps.at(49);
  dv_163 = 2 * d_124 * dv_133 - 4 * dv_79;
  DataVector& dv_164 = temps.at(56);
  dv_164 = (1.0 / 4.0) * dv_33 * dv_74;

  get(get<CurvedScalarWave::Tags::Psi>(*result)) =
      dv_33 * (-1.0 / 4.0 * d_124 * dv_73 * 1.0 / dv_31 + 1);
  get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(*result)) =
      (1.0 / 8.0) * dv_74 *
      (d_277 * dv_32 * 1.0 / dv_78 *
           (-d_165 * dv_139 - d_193 * dv_146 - d_194 * dv_148 -
            dv_79 *
                (-dv_129 *
                     (d_126 * d_183 * rp *
                          (d_108 * dv_131 + dv_52 * dv_55 + dv_71) +
                      4 * d_135 * d_18 * d_25 * dv_67 * rp +
                      4 * d_135 * d_25 * dv_52 * dv_54 * rp +
                      4 * d_18 * d_25 * dv_149 * dv_39 * rp +
                      4 * d_196 * d_25 * dv_39 * dv_52 * rp -
                      d_196 * d_50 * dv_67 +
                      2 * d_25 * dv_131 *
                          (-d_185 * zpddot +
                           d_216 * d_25 * d_61 * d_62 * xpdot +
                           d_219 * d_25 * d_61 * d_62 * ypdot +
                           d_229 * d_25 * zpdot - d_5 * d_67 * xpddot -
                           d_67 * d_87 * ypddot) -
                      d_264 * dv_152 - d_264 * dv_159 - d_50 * dv_149 * dv_54 -
                      d_50 * dv_52 *
                          (-d_110 * dv_110 + d_110 * dv_111 -
                           d_111 * (dv_113 - dv_114) - d_114 * dv_113 +
                           d_114 * dv_114 - d_115 * dv_110 + d_115 * dv_111 -
                           d_117 * dv_115 + d_117 * dv_116 +
                           d_120 * (dv_115 - dv_116) + d_155 * d_167 +
                           d_160 * d_165 + d_170 * d_173 + d_181 +
                           d_265 * dv_160 - d_265 * dv_161 - d_266 * dv_160 +
                           d_266 * dv_161 + d_269 * dv_6 - d_269 * dv_7 +
                           d_270 * dv_0 - d_270 * dv_1 - d_274 * dv_6 +
                           d_274 * dv_7 - d_276 * dv_0 + d_276 * dv_1) -
                      dv_150 * dv_53) +
                 dv_150 * (dv_125 + dv_67 * rp) +
                 dv_69 * (-d_135 * d_25 * dv_135 + dv_152 + 2 * dv_159) +
                 dv_126 * rp *
                     (d_183 * dv_128 * 1.0 / d_195 -
                      2 * d_195 *
                          (-d_140 * d_190 * dv_151 + d_182 * dv_149 +
                           d_196 * dv_127 + dv_121)) /
                     pow(d_125, 3.0 / 2.0)) +
            dv_80 * rp * (-dv_130 + dv_52 * (dv_125 + dv_68)) *
                (dv_122 - dv_123 + dv_81 + dv_85 - dv_86 - dv_88 - dv_89 +
                 dv_91 - dv_92 + dv_93 + dv_94 + dv_96 - dv_97 + dv_99)) -
       d_43 * dv_124 * dv_33 * dv_73 + 4 * dv_124 * dv_32);
  get<0>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      dv_164 *
      (d_124 * dv_146 + dv_163 * (d_37 * dv_20 + dv_105 -
                                  dv_162 * (d_168 + d_37 * (d_102 + d_16))));
  get<1>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      dv_164 *
      (d_124 * dv_148 + dv_163 * (2 * Dy * M * d_20 * d_25 * d_38 * d_39 + Dy -
                                  d_124 * d_32 * dv_11 - d_9 * dv_107 -
                                  dv_162 * (-d_103 * d_32 * d_35 +
                                            d_171 * d_38 * d_39 + ypdot)));
  get<2>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      dv_164 *
      (d_277 * dv_139 + dv_163 * (-d_164 * dv_47 + dv_102 -
                                  dv_162 * (d_162 * (d_102 + d_161) + d_163)));
}
}  // namespace CurvedScalarWave::Worldtube
