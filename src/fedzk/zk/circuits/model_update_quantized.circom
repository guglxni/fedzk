pragma circom 2.0.0;

/**
 * ModelUpdateQuantized - Circuit that handles quantized floating-point gradients
 *
 * This circuit accepts quantized gradients (integers representing scaled floats)
 * and provides cryptographic verification of gradient norms with quantization support.
 */

template ModelUpdateQuantized(n, scale_factor) {
    signal input quantized_gradients[n];  // Quantized gradient values
    signal input scale_factor_input;      // Scale factor used for quantization
    signal output original_norm;          // Norm in original scale
    signal output quantized_norm;         // Norm in quantized scale
    signal output gradient_count;         // Number of gradients processed

    signal scaled_gradients[n];           // Gradients in original scale
    signal sq[n];                         // Squared gradients
    signal acc[n+1];                      // Accumulator for norm calculation

    // Verify scale factor consistency
    scale_factor_input === scale_factor;

    // Initialize accumulator
    acc[0] <== 0;

    // Convert quantized gradients back to original scale and compute norm
    for (var i = 0; i < n; i++) {
        // Convert quantized value back to original scale
        scaled_gradients[i] <== quantized_gradients[i] * scale_factor_input;

        // Compute square
        sq[i] <== scaled_gradients[i] * scaled_gradients[i];

        // Accumulate
        acc[i+1] <== acc[i] + sq[i];
    }

    // Output results
    original_norm <== acc[n];
    quantized_norm <== acc[n] / (scale_factor * scale_factor);
    gradient_count <== n;

    // Constraint to ensure gradient_count is correct
    gradient_count === n;
}

/**
 * SecureModelUpdateQuantized - Enhanced version with security constraints
 */
template SecureModelUpdateQuantized(n, scale_factor, max_norm, min_active) {
    signal input quantized_gradients[n];
    signal input scale_factor_input;
    signal input max_norm_input;          // Maximum allowed norm
    signal input min_active_input;        // Minimum active gradients
    signal output original_norm;
    signal output quantized_norm;
    signal output gradient_count;
    signal output active_count;           // Number of non-zero gradients
    signal output norm_valid;             // Whether norm is within bounds
    signal output activity_valid;         // Whether activity meets minimum

    signal scaled_gradients[n];
    signal sq[n];
    signal acc[n+1];
    signal active[n+1];
    signal is_nonzero[n];

    // Verify input constraints
    scale_factor_input === scale_factor;
    max_norm_input === max_norm;
    min_active_input === min_active;

    // Initialize accumulators
    acc[0] <== 0;
    active[0] <== 0;

    for (var i = 0; i < n; i++) {
        // Convert quantized value back to original scale
        scaled_gradients[i] <== quantized_gradients[i] * scale_factor_input;

        // Compute square
        sq[i] <== scaled_gradients[i] * scaled_gradients[i];

        // Accumulate norm
        acc[i+1] <== acc[i] + sq[i];

        // Check if gradient is non-zero (approximate check for quantized values)
        is_nonzero[i] <== quantized_gradients[i] != 0 ? 1 : 0;
        active[i+1] <== active[i] + is_nonzero[i];
    }

    // Set outputs
    original_norm <== acc[n];
    quantized_norm <== acc[n] / (scale_factor * scale_factor);
    gradient_count <== n;
    active_count <== active[n];

    // Validate constraints
    norm_valid <== original_norm <= max_norm_input ? 1 : 0;
    activity_valid <== active_count >= min_active_input ? 1 : 0;

    // Enforce constraints
    norm_valid === 1;
    activity_valid === 1;
}

// Circuit instances with different configurations
component main { public [quantized_gradients, scale_factor_input] } = ModelUpdateQuantized(4, 1000);
component main_secure { public [quantized_gradients, scale_factor_input, max_norm_input, min_active_input] } = SecureModelUpdateQuantized(4, 1000, 1000000, 1);

