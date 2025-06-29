pragma circom 2.0.0;

// Simplified batch verification circuit for testing
template BatchVerificationSimple(batchSize, gradSize) {
    signal input gradientBatch[batchSize][gradSize];
    signal output aggregatedNorm;
    
    // Pre-declare all signals
    signal gradSq[batchSize][gradSize];
    signal gradAcc[batchSize][gradSize + 1];
    signal acc[batchSize + 1];
    
    // Initialize accumulator
    acc[0] <== 0;
    
    // For each gradient set in the batch
    for (var b = 0; b < batchSize; b++) {
        gradAcc[b][0] <== 0;
        
        // Calculate norm for this gradient set
        for (var i = 0; i < gradSize; i++) {
            gradSq[b][i] <== gradientBatch[b][i] * gradientBatch[b][i];
            gradAcc[b][i + 1] <== gradAcc[b][i] + gradSq[b][i];
        }
        
        // Accumulate norms
        acc[b + 1] <== acc[b] + gradAcc[b][gradSize];
    }
    
    // Final output
    aggregatedNorm <== acc[batchSize];
}

component main { public [gradientBatch] } = BatchVerificationSimple(4, 4);
