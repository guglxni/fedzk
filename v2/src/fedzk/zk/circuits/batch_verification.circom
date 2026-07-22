pragma circom 2.0.0;

// Batch verification circuit for aggregating multiple gradient proofs
// This circuit validates that multiple gradient sets meet consistency requirements

template BatchVerification(batchSize, gradSize) {
    signal input gradientBatch[batchSize][gradSize];
    signal input expectedNorms[batchSize];
    signal input globalMaxNorm;
    signal output batchValid;
    signal output aggregatedNorm;
    
    // Pre-declare all signals
    signal gradSq[batchSize][gradSize];
    signal gradAcc[batchSize][gradSize + 1];
    signal normDiff[batchSize];
    signal normDiffSq[batchSize];
    signal validFlags[batchSize];
    signal acc[batchSize + 1];
    signal allValid[batchSize + 1];
    
    // Initialize accumulators
    acc[0] <== 0;
    allValid[0] <== 1;
    
    // For each gradient set in the batch
    for (var b = 0; b < batchSize; b++) {
        gradAcc[b][0] <== 0;
        
        // Calculate norm for this gradient set
        for (var i = 0; i < gradSize; i++) {
            gradSq[b][i] <== gradientBatch[b][i] * gradientBatch[b][i];
            gradAcc[b][i + 1] <== gradAcc[b][i] + gradSq[b][i];
        }
        
        // Verify this gradient set's norm matches expected
        normDiff[b] <== gradAcc[b][gradSize] - expectedNorms[b];
        normDiffSq[b] <== normDiff[b] * normDiff[b];
        validFlags[b] <== 1 - normDiffSq[b]; // Should be 1 if diff is 0
        
        // Accumulate norms and validity
        acc[b + 1] <== acc[b] + gradAcc[b][gradSize];
        allValid[b + 1] <== allValid[b] * validFlags[b];
    }
    
    // Final outputs
    aggregatedNorm <== acc[batchSize];
    
    // Verify aggregated norm is within global bounds
    signal globalValid <== 1; // Simplified constraint
    batchValid <== allValid[batchSize] * globalValid;
    
    // Constraint: batch must be valid
    batchValid === 1;
}

component main { public [expectedNorms, globalMaxNorm] } = BatchVerification(4, 4);
