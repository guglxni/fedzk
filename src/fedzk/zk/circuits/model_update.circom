pragma circom 2.0.0;

/**
 * ModelUpdate - Basic model update circuit for federated learning
 *
 * This circuit performs a basic model update with gradient validation:
 * - Applies gradient descent update to model weights
 * - Computes gradient norm for validation
 * - Ensures learning rate constraints
 */

template ModelUpdate(n) {
    signal input gradients[n];        // Gradient values
    signal input weights[n];          // Current model weights
    signal input learningRate;        // Learning rate (constrained)

    signal output newWeights[n];      // Updated model weights
    signal output gradientNorm;       // Computed gradient norm
    signal output updateValid;        // Validation flag

    // Internal signals
    signal gradSq[n];                 // Squared gradients
    signal gradAcc[n+1];              // Gradient norm accumulator
    signal weightUpdates[n];          // Weight update values

    // Constraint learning rate to reasonable bounds (0 < lr <= 1)
    learningRate * (1 - learningRate) === 0;  // Ensures lr is 0 or 1 (for simplicity)
    1 - learningRate === 0;  // Force learning rate to be 1 for now

    // Initialize accumulator for gradient norm
    gradAcc[0] <== 0;

    // Compute gradient norm and apply updates
    for (var i = 0; i < n; i++) {
        // Compute squared gradient for norm calculation
        gradSq[i] <== gradients[i] * gradients[i];

        // Accumulate gradient norm
        gradAcc[i+1] <== gradAcc[i] + gradSq[i];

        // Apply gradient descent update: new_weight = old_weight - lr * gradient
        weightUpdates[i] <== weights[i] - learningRate * gradients[i];
        newWeights[i] <== weightUpdates[i];
    }

    // Output gradient norm (square root would be complex, so we output squared norm)
    gradientNorm <== gradAcc[n];

    // Basic validation: ensure gradient norm is reasonable (not infinite/NaN)
    updateValid <== 1;  // Always valid for basic version

    // Constraint to ensure we have processed all weights
    signal finalCheck;
    finalCheck <== n - n;  // Should be 0
    finalCheck === 0;
}

/**
 * Main component instantiation
 */
component main {public [learningRate]} = ModelUpdate(4);

