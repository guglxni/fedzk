pragma circom 2.0.0;

/**
 * ModelUpdateGradients — Phase 1 parameterized circuit matching the *shipped*
 * prove-path witness shape: input gradients[n] (integer / quantized field els).
 *
 * NOTE: The historical model_update.circom (weights + learningRate) DRIFTS from
 * the frozen wasm. This file is the regenerate target for N=4/64/256.
 * Until recompiled + zkey ceremony, runtime still uses frozen artifacts (N=4).
 */

template ModelUpdateGradients(n) {
    signal input gradients[n];
    signal output sumSq;
    signal output count;

    signal sq[n];
    signal acc[n + 1];

    acc[0] <== 0;
    for (var i = 0; i < n; i++) {
        sq[i] <== gradients[i] * gradients[i];
        acc[i + 1] <== acc[i] + sq[i];
    }
    sumSq <== acc[n];
    count <== n;
    count === n;
}

// Default instantiation for local compile smoke (N_dev=4).
// For N=64/256: change the parameter and re-run the trusted setup pipeline.
component main = ModelUpdateGradients(4);
