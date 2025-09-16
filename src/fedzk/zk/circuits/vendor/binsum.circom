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

// Template for binary summation of multiple inputs
template BinSum(n, m) {
    signal input in[m][n];
    signal output out[n];

    var lin = 0;
    var lout = 0;

    var k;
    var j;

    for (k=0; k<n; k++) {
        lin = 0;
        for (j=0; j<m; j++) {
            lin += in[j][k];
        }

        for (j=0; j<n; j++) {
            if (j < k) {
                lin += out[j] * (1 << j);
            }
        }

        out[k] <-- lin \ (1 << k);
        lout += out[k] * (1 << k);

        out[k] * (1 << k) <= lin;
        lin - out[k] * (1 << k) < (1 << k);
    }

    lout === lin;
}

// Template for summing multiple numbers
template Sum(n) {
    signal input in[n];
    signal output out;

    signal sums[n];
    sums[0] <== in[0];

    for (var i = 1; i < n; i++) {
        sums[i] <== sums[i-1] + in[i];
    }

    out <== sums[n-1];
}

// Template for calculating the sum of squares
template SumOfSquares(n) {
    signal input in[n];
    signal output out;

    signal squares[n];
    signal sums[n];

    squares[0] <== in[0] * in[0];
    sums[0] <== squares[0];

    for (var i = 1; i < n; i++) {
        squares[i] <== in[i] * in[i];
        sums[i] <== sums[i-1] + squares[i];
    }

    out <== sums[n-1];
}

