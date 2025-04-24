pragma circom 2.0.0;

template ModelUpdate(n) {
    signal input gradients[n];
    signal output norm;
    signal acc[n+1];
    signal sq[n];

    // Initialize accumulator
    acc[0] <== 0;

    // Compute squares and accumulate step by step
    for (var i = 0; i < n; i++) {
        sq[i] <== gradients[i] * gradients[i];
        acc[i+1] <== acc[i] + sq[i];
    }

    // Output the final accumulated sum
    norm <== acc[n];
}

component main { public [gradients] } = ModelUpdate(4); 