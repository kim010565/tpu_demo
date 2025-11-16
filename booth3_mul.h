#pragma once

#include "systemc.h"

// booth3乘法(Radix-8)，对于B进行拆分重组：参考https://zhuanlan.zhihu.com/p/1890916925300769414
// 对应《FAST MULTIPLICATION ALGORITHMS AND IMPLEMENTATION.pdf》中的Figure 2.7: 16 bit Booth 3 multiply; 推导过程中的b[-1]对应LSB补0

// 模拟bf16(1-8-7)中的乘法器
// 被乘数a和乘数b的符号位分别提取出来，记为s_a和s_b
// 被乘数a和乘数b的尾数位分别提取出来，记为a和b（包含隐藏位）
// c = (s_a*s_b)*a*b, 17bit = 1bit + 8bit + 8bit
inline sc_uint<17> booth3_mul(sc_uint<1> s_a, sc_uint<1> s_b, sc_uint<8> a, sc_uint<8> b) {
  sc_uint<17> c;
  sc_uint<10> a3 = (a << 1) + a;    // a * 3
  sc_uint<1> res_sign = s_a ^ s_b;  // 通过符号位来翻转booth_code来实现*1或*-1
  sc_uint<4> booth_code[3];         // 用于部分积查表的Multiplier Bits
  sc_uint<10> booth_pp[3];          // 部分积结果
  sc_uint<1> booth_inv[3];          // 对应补码取反后+1部分
  sc_uint<17> pp[4];                // 华莱士压缩结果
  sc_uint<17> sum0, sum1;           // 用于4-2压缩
  sc_uint<17> carry0, carry1;
  booth_code[0] = (b[2], b[1], b[0], sc_uint<1>(0)) ^ (res_sign, res_sign, res_sign, res_sign);
  booth_code[1] = (b[5], b[4], b[3], b[2]) ^ (res_sign, res_sign, res_sign, res_sign);
  booth_code[2] = (sc_uint<1>(0), b[7], b[6], b[5]) ^ (res_sign, res_sign, res_sign, res_sign);
  sc_uint<8> not_a = ~a;
  sc_uint<10> not_a3 = ~a3;
  for (int i = 0; i < 3; i++) {
    switch (booth_code[i]) {
      case 0:
      case 15:  // +/- 0*a
        booth_pp[i] = 0;
        booth_inv[i] = 0;
        break;
      case 1:
      case 2:  // +1*a
        booth_pp[i] = (sc_uint<1>(0), sc_uint<1>(0), a);
        booth_inv[i] = 0;
        break;
      case 3:
      case 4:  // +2*a
        booth_pp[i] = (sc_uint<1>(0), a, sc_uint<1>(0));
        booth_inv[i] = 0;
        break;
      case 5:
      case 6:  // +3*a
        booth_pp[i] = a3;
        booth_inv[i] = 0;
        break;
      case 7:  // +4*a
        booth_pp[i] = (a, sc_uint<1>(0), sc_uint<1>(0));
        booth_inv[i] = 0;
        break;
      case 8:  // -4*a
        booth_pp[i] = (not_a, sc_uint<1>(1), sc_uint<1>(1));
        booth_inv[i] = 1;
        break;
      case 9:
      case 10:  // -3*a
        booth_pp[i] = not_a3;
        booth_inv[i] = 1;
        break;
      case 11:
      case 12:  // -2*a
        booth_pp[i] = (sc_uint<1>(1), not_a, sc_uint<1>(1));
        booth_inv[i] = 1;
        break;
      default:  // -1*a
        booth_pp[i] = (sc_uint<1>(1), sc_uint<1>(1), not_a);
        booth_inv[i] = 1;
        break;
    }
  }
  pp[0] = (sc_uint<3>(0), !booth_inv[0], booth_inv[0], booth_inv[0], booth_inv[0], booth_pp[0]);                   // 0 0 0 S_bar S S S PP (3+4+10=17)
  pp[1] = (sc_uint<1>(0), sc_uint<1>(1), sc_uint<1>(1), !booth_inv[1], booth_pp[1], sc_uint<2>(0), booth_inv[0]);  // 0 1 1 S_bar PP 0 0 S (1+3+10+3=17bit)
  pp[2] = (!booth_inv[2], booth_pp[2], sc_uint<2>(0), booth_inv[1], sc_uint<3>(0));                                // S_bar PP 0 S 0 0 0 (1+10+6=17bit)
  pp[3] = (sc_uint<8>(0), sc_uint<2>(0), booth_inv[2], sc_uint<6>(0));                                             // 0... 0 0 S 0 0 0 0 0 0 (8+9=17bit)
  // 两次3-2压缩组成一次4-2压缩
  sum0 = pp[0] ^ pp[1] ^ pp[2];
  carry0 = ((pp[0] & pp[1]) ^ (pp[2] & (pp[0] ^ pp[1]))) << 1;
  sum1 = sum0 ^ carry0 ^ pp[3];
  carry1 = ((sum0 & carry0) ^ (pp[3] & (sum0 ^ carry0))) << 1;
  // adder
  c = (sum1 + carry1);

  return c;
}
