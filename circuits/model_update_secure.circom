pragma circom 2.0.0;

// Custom IsZero gadget for non-zero detection
template IsZero() {
    signal input in;
    signal output out;

    signal inv;
    // inv = 1/in when in != 0, else inv = 0
    inv <-- in != 0 ? 1/in : 0;

    // out = 1 - in * inv
    out <== -in * inv + 1;
    // enforce in * out == 0
    in * out === 0;
}

template ModelUpdateSecure(n) {
    signal input gradients[n];
    signal input maxNorm;
    signal input minNonZero;
    signal output norm;
    signal output nonZeroCount;
    signal acc[n+1];
    signal sq[n];
    signal count[n+1];
    signal isNZ[n];
    component iz[n];

    // Initialize accumulators
    acc[0] <== 0;
    count[0] <== 0;

    // Compute squares and nonzero count stepwise
    for (var i = 0; i < n; i++) {
        sq[i] <== gradients[i] * gradients[i];
        acc[i+1] <== acc[i] + sq[i];

        // Use custom IsZero gadget for non-zero detection
        iz[i] = IsZero();
        iz[i].in <== gradients[i];
        isNZ[i] <== 1 - iz[i].out;
        count[i+1] <== count[i] + isNZ[i];
    }

    // Assign outputs and enforce dynamic bounds
    norm <== acc[n];
    nonZeroCount <== count[n];
    assert(norm <= maxNorm);
    assert(nonZeroCount >= minNonZero);
}

// Expose gradients and dynamic bounds as public inputs
component main { public [gradients, maxNorm, minNonZero] } = ModelUpdateSecure(4); 