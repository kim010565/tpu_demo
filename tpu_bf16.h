#pragma once

#include "booth3_mul.h"

// bf16, 1-8-7：1bit符号位，8bit指数位，7bit尾数位
typedef sc_uint<16> BF16;
// fp32, 1-8-23：1bit符号位，8bit指数位，23bit尾数位
typedef sc_uint<32> FP32;
// 自定义中间psum存储格式：8bit指数位，27bit尾数位（尾数包含符号位，用补码表示）。浮点值=（尾数/2^25）*2^(指数-127)
// 27bit中：1bit对应符号位；1bit对应隐藏位；剩下25bit对应IEEE 754 FP32的23bit尾数。即留了2bit用来处理舍入操作。
// 自定义psum的NAN：指数全1，尾数低26bit全为1
// 自定义psum的INF：指数全1，尾数低26bit全为0，第26bit用于表示+/-INF
typedef sc_uint<35> FPSUM;

// 统计前导0的个数
inline sc_uint<5> leading_zero(sc_uint<27> x) {
  for (int i = 0; i < 26; i++) {
    if (0 != x[26 - i]) {
      return sc_uint<5>(i);
    }
  }
  return sc_uint<5>(26);
}

// 统计前导符号位的个数
template <int N>
inline sc_uint<6> leading_sign(sc_uint<N> x) {
  for (int i = 0; i < (N - 1); i++) {
    if (x[N - 1] != x[N - 2 - i]) {
      return sc_uint<6>(i);
    }
  }
  return sc_uint<6>(N - 1);
}

// FP32转自定义psum
inline FPSUM convert_FP32_to_FPSUM(FP32 a) {
  FPSUM res;
  sc_uint<8> fp32_exp = a.range(30, 23);        // 取FP32的指数位
  sc_uint<23> fp32_fraction = a.range(22, 0);   // 取FP32的尾数位
  if (fp32_exp == 255 && fp32_fraction != 0) {  // 判断FP32输入是否是NAN
    res = (sc_uint<8>(255), sc_uint<27>(0x3FFFFFF));
  } else if (fp32_exp == 255 && fp32_fraction == 0) {  // 判断FP32输入是否是INF
    res = (sc_uint<8>(255), sc_uint<1>(a[31]), sc_uint<26>(0));
  } else if (fp32_exp == 0 && fp32_fraction == 0) {  // 判断FP32输入是否是0
    res = (sc_uint<8>(0), sc_uint<27>(0));
  } else if (fp32_exp == 0) {                                                                             // 判断FP32输入是否是非规格化浮点数
    res = (fp32_exp, sc_uint<27>(sc_int<24>((1 - 2 * a[31]) * fp32_fraction) << 3));                      // (1bit符号+23bit尾数) << 3 => 27bit自定义psum尾数
  } else {                                                                                                // 判断FP32输入是否是规格化浮点数
    res = (fp32_exp, sc_uint<27>(((1 - 2 * a[31]) * sc_uint<24>((sc_uint<1>(1), fp32_fraction))) << 2));  // (1bit符号+1bit隐藏+23bit尾数) << 2 => 27bit自定义psum尾数
  }
  return res;
}

// 自定义psum转FP32
// round_mod: 0-rounds to nearest even，1-rounds towards zero，2-rounds towards negative infinity, 3-rounds towards positive infinity
inline FP32 convert_FPSUM_to_FP32(FPSUM psum, uint8_t round_mod = 0) {
  FP32 res;
  sc_uint<8> psum_exp = psum.range(34, 27);                          // 取自定义psum的指数位
  sc_uint<27> psum_fraction = sc_uint<27>(psum.range(26, 0));        // 取自定义psum的尾数位
  if (psum_exp == 255 && psum_fraction.range(25, 0) != 0) {          // 判断psum是否是NAN
    res = (sc_uint<1>(0), sc_uint<8>(0xFF), sc_uint<23>(0x7FFFFF));  // 参考NV，输出仅支持QNAN（不支持SNAN）
  } else if (psum_exp == 255 && psum_fraction.range(25, 0) == 0) {   // 判断psum是否是INF
    res = (sc_uint<1>(psum_fraction[26]), sc_uint<8>(0xFF), sc_uint<23>(0x0));
  } else if (psum_fraction == 0) {  // 判断psum是否是0
    res = 0x0;
  } else {
    sc_uint<1> sign = sc_uint<1>(psum_fraction[26]);  // 取自定义psum的符号位
    sc_uint<27> psum_fraction_abs = psum_fraction;
    if (sign == 1) {
      psum_fraction_abs = ~psum_fraction_abs + 1;  // 自定义psum尾数取abs得到原码
    }
    sc_uint<5> leading = leading_zero(psum_fraction_abs);  // 统计psum_fraction_abs前导0的个数
    psum_fraction_abs = psum_fraction_abs << leading;      // psum_fraction_abs顶格存放
    sc_int<9> exp_tmp = psum_exp - leading + 1;            // 折算等效FP32指数位数值
    if (exp_tmp > 0) {                                     // 输出为规格化浮点数
      sc_uint<1> guard_bit = sc_uint<1>(psum_fraction_abs[3]);
      sc_uint<1> round_bit = sc_uint<1>(psum_fraction_abs[2]);
      sc_uint<1> sticky_bit = sc_uint<1>(psum_fraction_abs[1]) | sc_uint<1>(psum_fraction_abs[0]);
      sc_uint<23> fp32_fraction = psum_fraction_abs.range(25, 3);  // psum_fraction_abs顶格存放后，最高bit对应隐藏位，低3bit用于舍入操作
      // 根据round_mod对尾数做舍入操作
      if (0 == round_mod) {  // rounds to nearest even
        if ((round_bit & sticky_bit) | (round_bit & guard_bit)) {
          if (fp32_fraction == 0x7FFFFF) {
            fp32_fraction = 0x0;
            exp_tmp = exp_tmp + 1;
          } else {
            fp32_fraction = fp32_fraction + 1;
          }
        }
      } else if (1 == round_mod) {  // rounds towards zero
        // do nothing
      } else if (2 == round_mod) {  // rounds towards negative infinity
        if (1 == sign && 1 == (round_bit | sticky_bit)) {
          if (fp32_fraction == 0x7FFFFF) {
            fp32_fraction = 0x0;
            exp_tmp = exp_tmp + 1;
          } else {
            fp32_fraction = fp32_fraction + 1;
          }
        }
      } else {  // rounds towards positive infinity
        if (0 == sign && 1 == (round_bit | sticky_bit)) {
          if (fp32_fraction == 0x7FFFFF) {
            fp32_fraction = 0x0;
            exp_tmp = exp_tmp + 1;
          } else {
            fp32_fraction = fp32_fraction + 1;
          }
        }
      }
      res = (sign, sc_uint<8>(exp_tmp.range(7, 0)), fp32_fraction);
    } else {                                                           // 输出为非规格化浮点数
      sc_uint<23> fp32_fraction = psum_fraction_abs >> (4 - exp_tmp);  // 计算非规格化尾数
      sc_uint<1> guard_bit = sc_uint<1>(fp32_fraction[0]);
      sc_uint<1> round_bit = sc_uint<1>((psum_fraction_abs >> (3 - exp_tmp)) & 1);
      sc_uint<1> sticky_bit = ((psum_fraction_abs & ((1 << (3 - exp_tmp)) - 1)) == 0) ? 0 : 1;
      // 根据round_mod对尾数做舍入操作
      exp_tmp = 0;
      if (0 == round_mod) {  // rounds to nearest even
        if ((round_bit & sticky_bit) | (round_bit & guard_bit)) {
          if (fp32_fraction == 0x7FFFFF) {
            fp32_fraction = 0x0;
            exp_tmp = exp_tmp + 1;
          } else {
            fp32_fraction = fp32_fraction + 1;
          }
        }
      } else if (1 == round_mod) {  // rounds towards zero
        // do nothing
      } else if (2 == round_mod) {  // rounds towards negative infinity
        if (1 == sign && 1 == (round_bit | sticky_bit)) {
          if (fp32_fraction == 0x7FFFFF) {
            fp32_fraction = 0x0;
            exp_tmp = exp_tmp + 1;
          } else {
            fp32_fraction = fp32_fraction + 1;
          }
        }
      } else {  // rounds towards positive infinity
        if (0 == sign && 1 == (round_bit | sticky_bit)) {
          if (fp32_fraction == 0x7FFFFF) {
            fp32_fraction = 0x0;
            exp_tmp = exp_tmp + 1;
          } else {
            fp32_fraction = fp32_fraction + 1;
          }
        }
      }
      res = (sign, sc_uint<8>(exp_tmp.range(7, 0)), fp32_fraction);
    }
  }
  return res;
}

// K=16 & psum => 累加导致的高位扩展比特=5
#define EXPAND_BIT_BF16 5
// "mma_precision_test"测得NV MMA(BF16)能容忍的指数最大差异为25，即尾数对阶移位后至少需要26bit才能保证容忍最大2^25与最小2^0
// 因此，低位额外保留比特=26-7-7-1=11
#define EXTRA_BIT_BF16 11
// 仅模拟1*16*1的乘加操作(K=16方向乘累加)，不涉及M、N循环
// D = A*B + PSUM
// EXPAND_BIT表示累加导致的高位扩展比特
// EXTRA_BIT表示对阶移位后低位额外保留比特
template <int EXPAND_BIT, int EXTRA_BIT>
inline FPSUM bf16_1x16x1(BF16 a[16], BF16 b[16], FPSUM psum_c, uint8_t round_mod = 0) {
  bool nan_flag = false;
  bool posinf_flag = false;
  bool neginf_flag = false;
  sc_uint<1> a_sign[16];
  sc_uint<8> a_exp[16];
  sc_uint<7> a_fraction[16];
  sc_uint<1> b_sign[16];
  sc_uint<8> b_exp[16];
  sc_uint<7> b_fraction[16];
  FPSUM psum_d;

  for (int k = 0; k < 16; k++) {
    a_sign[k] = a[k].range(15, 15);    // 取BF16(a)的符号位
    a_exp[k] = a[k].range(14, 7);      // 取BF16(a)的指数位
    a_fraction[k] = a[k].range(6, 0);  // 取BF16(a)的尾数位
    b_sign[k] = b[k].range(15, 15);    // 取BF16(b)的符号位
    b_exp[k] = b[k].range(14, 7);      // 取BF16(b)的指数位
    b_fraction[k] = b[k].range(6, 0);  // 取BF16(b)的尾数位
    // 特殊情况判断
    if ((a_exp[k] == 255 && a_fraction[k] == 0 && b_exp[k] == 0 && b_fraction[k] == 0) ||  // a==inf && b==0
        (a_exp[k] == 0 && a_fraction[k] == 0 && b_exp[k] == 255 && b_fraction[k] == 0) ||  // a==0 && b==inf
        (a_exp[k] == 255 && a_fraction[k] != 0) ||                                         // a==nan
        (b_exp[k] == 255 && b_fraction[k] != 0)) {                                         // b==nan
      nan_flag = true;
    } else if ((a_exp[k] == 255 && a_fraction[k] == 0) || (b_exp[k] == 255 && b_fraction[k] == 0)) {  // a==inf || b==inf
      if (a_sign[k] == b_sign[k]) {
        posinf_flag = true;
      } else {
        neginf_flag = true;
      }
    }
  }
  sc_uint<8> c_exp = psum_c.range(34, 27);                    // 取FPSUM(c)的指数位
  sc_uint<27> c_fraction = sc_uint<27>(psum_c.range(26, 0));  // 取FPSUM(c)的尾数位; c浮点值=c_fraction*2^(c_exp-127-25)
  if (c_exp == 255 && c_fraction.range(25, 0) != 0) {         // c==nan
    nan_flag = true;
  } else if (c_exp == 255 && c_fraction.range(25, 0) == 0) {  // c==inf
    if (c_fraction == 0) {
      posinf_flag = true;
    } else {
      neginf_flag = true;
    }
  }

  if (nan_flag || (posinf_flag && neginf_flag)) {  // nan
    psum_d = (sc_uint<8>(255), sc_uint<27>(0x3FFFFFF));
  } else if (posinf_flag) {  // +inf
    psum_d = (sc_uint<8>(255), sc_uint<27>(0x0));
  } else if (neginf_flag) {  // -inf
    psum_d = (sc_uint<8>(255), sc_uint<27>(0x4000000));
  } else {  // 正常情况
    // 指数拉齐
    sc_int<10> ab_exp[16];
    sc_uint<9> ab_shift[16];
    // (a_exp-127-7)+(b_exp-127-7)-EXTRA_BIT与c_exp-127-25-(17+EXTRA_BIT-27)拉齐 => a_exp+b_exp与c_exp-127-25-(17+EXTRA_BIT-27)+127*2+7*2+EXTRA_BIT拉齐
    sc_int<10> c_exp_cvt = c_exp - 127 - 25 - (17 + EXTRA_BIT - 27) + 127 * 2 + 7 * 2 + EXTRA_BIT;  // psum_c exp对阶折算到a*b对阶移位结果
    sc_uint<9> c_shift_cvt;
    sc_int<10> max_exp = c_exp_cvt;
    for (int k = 0; k < 16; k++) {
      if (!((a_exp[k] == 0 && a_fraction[k] == 0) || (b_exp[k] == 0 && b_fraction[k] == 0))) {  // !(a==0 || b==0)
        ab_exp[k] = ((a_exp[k] == 0) ? sc_uint<8>(1) : a_exp[k]) + ((b_exp[k] == 0) ? sc_uint<8>(1) : b_exp[k]);
      } else {
        ab_exp[k] = 0;
      }
      max_exp = std::max(max_exp, ab_exp[k]);
    }
    c_shift_cvt = max_exp - c_exp_cvt;
    for (int k = 0; k < 16; k++) {
      ab_shift[k] = max_exp - ab_exp[k];
    }

    // 尾数相乘 & 对阶移位
    sc_uint<8> a_mantissa[16];
    sc_uint<8> b_mantissa[16];
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> ab_mantissa[16];  // bf16尾数=7+1=8；尾数乘法结果=8(a)+8(b)+1(符号位)=17
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> c_mantissa_cvt;
    if constexpr ((17 + EXTRA_BIT) >= 27) {
      c_mantissa_cvt = sc_int<17 + EXPAND_BIT + EXTRA_BIT>(sc_int<27>(c_fraction)) << (17 + EXTRA_BIT - 27);
    } else {
      c_mantissa_cvt = sc_int<17 + EXPAND_BIT + EXTRA_BIT>(sc_int<27>(c_fraction)) >> (27 - 17 - EXTRA_BIT);
    }
    for (int k = 0; k < 16; k++) {
      if (!((a_exp[k] == 0 && a_fraction[k] == 0) || (b_exp[k] == 0 && b_fraction[k] == 0))) {  // !(a==0 || b==0)
        a_mantissa[k] = ((a_exp[k] == 0) ? sc_uint<1>(0) : sc_uint<1>(1), a_fraction[k]);       // 补隐藏位；a浮点值=a_mantissa*2^(a_exp-127-7)
        b_mantissa[k] = ((b_exp[k] == 0) ? sc_uint<1>(0) : sc_uint<1>(1), b_fraction[k]);       // 补隐藏位；b浮点值=b_mantissa*2^(b_exp-127-7)
        sc_uint<17> ab_mantissa_tmp = booth3_mul(a_sign[k], b_sign[k], a_mantissa[k], b_mantissa[k]);
        if (ab_shift[k] > (17 + EXTRA_BIT)) {
          ab_mantissa[k] = 0;
        } else {
          ab_mantissa[k] = (sc_int<17 + EXPAND_BIT + EXTRA_BIT>(sc_int<17>(ab_mantissa_tmp)) << EXTRA_BIT) >> ab_shift[k];  // 根据MSB扩展高位后，再对阶移位
        }
      } else {
        ab_mantissa[k] = 0;
      }
    }
    if (c_shift_cvt > (17 + EXTRA_BIT)) {
      c_mantissa_cvt = 0;
    } else {
      c_mantissa_cvt = sc_int<17 + EXPAND_BIT + EXTRA_BIT>(c_mantissa_cvt) >> c_shift_cvt;  // 根据MSB扩展高位后，再对阶移位
    }

    // 对阶移位后的累加（华莱士加法树）
    //// Stage 0 (16 -> 8)
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> sum0 = ab_mantissa[0] ^ ab_mantissa[1] ^ ab_mantissa[2];
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> carry0 = ((ab_mantissa[0] & ab_mantissa[1]) ^ (ab_mantissa[2] & (ab_mantissa[0] ^ ab_mantissa[1]))) << 1;
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> sum1 = sum0 ^ carry0 ^ ab_mantissa[3];
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> carry1 = ((sum0 & carry0) ^ (ab_mantissa[3] & (sum0 ^ carry0))) << 1;

    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> sum2 = ab_mantissa[4] ^ ab_mantissa[5] ^ ab_mantissa[6];
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> carry2 = ((ab_mantissa[4] & ab_mantissa[5]) ^ (ab_mantissa[6] & (ab_mantissa[4] ^ ab_mantissa[5]))) << 1;
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> sum3 = sum2 ^ carry2 ^ ab_mantissa[7];
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> carry3 = ((sum2 & carry2) ^ (ab_mantissa[7] & (sum2 ^ carry2))) << 1;

    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> sum4 = ab_mantissa[8] ^ ab_mantissa[9] ^ ab_mantissa[10];
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> carry4 = ((ab_mantissa[8] & ab_mantissa[9]) ^ (ab_mantissa[10] & (ab_mantissa[8] ^ ab_mantissa[9]))) << 1;
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> sum5 = sum4 ^ carry4 ^ ab_mantissa[11];
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> carry5 = ((sum4 & carry4) ^ (ab_mantissa[11] & (sum4 ^ carry4))) << 1;

    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> sum6 = ab_mantissa[12] ^ ab_mantissa[13] ^ ab_mantissa[14];
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> carry6 = ((ab_mantissa[12] & ab_mantissa[13]) ^ (ab_mantissa[14] & (ab_mantissa[12] ^ ab_mantissa[13]))) << 1;
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> sum7 = sum6 ^ carry6 ^ ab_mantissa[15];
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> carry7 = ((sum6 & carry6) ^ (ab_mantissa[15] & (sum6 ^ carry6))) << 1;

    //// Stage 1 (8 -> 4)
    sum0 = sum1 ^ carry1 ^ sum3;
    carry0 = ((sum1 & carry1) ^ (sum3 & (sum1 ^ carry1))) << 1;
    sum1 = sum0 ^ carry0 ^ carry3;
    carry1 = ((sum0 & carry0) ^ (carry3 & (sum0 ^ carry0))) << 1;

    sum4 = sum5 ^ carry5 ^ sum7;
    carry4 = ((sum5 & carry5) ^ (sum7 & (sum5 ^ carry5))) << 1;
    sum5 = sum4 ^ carry4 ^ carry7;
    carry5 = ((sum4 & carry4) ^ (carry7 & (sum4 ^ carry4))) << 1;

    //// Stage 2 (4 -> 2)
    sum0 = sum1 ^ carry1 ^ sum5;
    carry0 = ((sum1 & carry1) ^ (sum5 & (sum1 ^ carry1))) << 1;
    sum1 = sum0 ^ carry0 ^ carry5;
    carry1 = ((sum0 & carry0) ^ (carry5 & (sum0 ^ carry0))) << 1;

    /// Stage 3 (add psum, 3 -> 2)
    sum0 = sum1 ^ carry1 ^ c_mantissa_cvt;
    carry0 = ((sum1 & carry1) ^ (c_mantissa_cvt & (sum1 ^ carry1))) << 1;

    // adder
    sc_uint<17 + EXPAND_BIT + EXTRA_BIT> reduced_rst = sum0 + carry0;

    // 规格化到自定义psum格式
    if (0 == reduced_rst) {
      psum_d = 0;
    } else {
      sc_uint<6> leading = leading_sign<17 + EXPAND_BIT + EXTRA_BIT>(reduced_rst);
      if ((-10 + EXPAND_BIT + EXTRA_BIT - leading.to_int()) >= 27) {
        psum_d = 0;
      } else {
        // 17 + EXPAND_BIT + EXTRA_BIT - 27 => -10 + EXPAND_BIT + EXTRA_BIT
        sc_uint<27> psum_d_fraction = sc_int<17 + EXPAND_BIT + EXTRA_BIT>(reduced_rst << leading) >> (-10 + EXPAND_BIT + EXTRA_BIT);
        sc_int<11> psum_d_exp = max_exp - leading + (-10 + EXPAND_BIT + EXTRA_BIT) + 127 - 127 * 2 + 25 - 7 * 2 - EXTRA_BIT;
        if (psum_d_exp < 0) {
          sc_int<11> shift_v = -psum_d_exp;
          if (shift_v >= 27) {
            shift_v = 27;
          }
          psum_d = (sc_uint<8>(0), sc_uint<27>((sc_int<27>(psum_d_fraction)) >> shift_v));
        } else if (psum_d_exp >= 255) {
          if (psum_d_fraction[26] == 0) {
            psum_d = (sc_uint<8>(255), sc_uint<27>(0x0));  // +inf
          } else {
            psum_d = (sc_uint<8>(255), sc_uint<27>(0x4000000));  // -inf
          }
        } else {
          psum_d = (psum_d_exp, psum_d_fraction);
        }
      }
    }
  }
  return psum_d;
}

template <int EXPAND_BIT, int EXTRA_BIT>
inline FP32 bf16_1x16x1(BF16 a[16], BF16 b[16], FP32 c, uint8_t round_mod = 0) {
  FPSUM psum_c = convert_FP32_to_FPSUM(c);  // 将FP32(c)转换为自定义psum格式
  FPSUM psum_d = bf16_1x16x1<EXPAND_BIT, EXTRA_BIT>(a, b, psum_c, round_mod);
  return convert_FPSUM_to_FP32(psum_d, round_mod);
}