/*
    Copyright 2018 0KIMS association.

    This file is part of circom (Zero Knowledge Circuit Compiler).

    circom is a free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    circom is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
    License for more details.

    You should have received a copy of the GNU General Public License
    along with circom. If not, see <https://www.gnu.org/licenses/>.
*/
pragma circom 2.0.0;

// Template to convert a number to its binary representation
template Num2Bits(n) {
    signal input in;
    signal output out[n];
    var lc1 = 0;

    var e2 = 1;
    for (var i = 0; i < n; i++) {
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] - 1) === 0;
        lc1 += out[i] * e2;
        e2 = e2 + e2;
    }

    lc1 === in;
}

// Template to convert a number to its binary representation (negative)
template Num2BitsNeg(n) {
    signal input in;
    signal output out[n];
    var lc1 = 0;

    var e2 = 1;
    for (var i = 0; i < n; i++) {
        out[i] <-- ((in + (1 << n)) >> i) & 1;
        out[i] * (out[i] - 1) === 0;
        lc1 += out[i] * e2;
        e2 = e2 + e2;
    }

    lc1 === in + (1 << n);
}

// Template to convert binary representation back to number
template Bits2Num(n) {
    signal input in[n];
    signal output out;
    var lc1 = 0;

    var e2 = 1;
    for (var i = 0; i < n; i++) {
        lc1 += in[i] * e2;
        e2 = e2 + e2;
    }

    out <== lc1;
}

// Template to convert binary representation back to number (with constraints)
template Bits2NumConstrained(n) {
    signal input in[n];
    signal output out;
    var lc1 = 0;

    var e2 = 1;
    for (var i = 0; i < n; i++) {
        in[i] * (in[i] - 1) === 0;  // Ensure each bit is 0 or 1
        lc1 += in[i] * e2;
        e2 = e2 + e2;
    }

    out <== lc1;
}

