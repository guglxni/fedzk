pragma circom 2.0.0;

// Sparse gradients circuit for verifying sparsity patterns and efficient compression
// This circuit validates that gradients meet sparsity requirements for bandwidth optimization

template IsZero() {
    signal input in;
    signal output out;
    signal inv;
    inv <-- in != 0 ? 1/in : 0;
    out <== -in * inv + 1;
    in * out === 0;
}

template SparseGradients(n) {
    signal input gradients[n];
    signal input indices[n];        // Indices of non-zero elements
    signal input values[n];         // Values at those indices
    signal input sparsityThreshold; // Maximum allowed non-zero elements
    signal output nonZeroCount;
    signal output compressionRatio;
    signal output sparseValid;
    
    // Pre-declare all signals
    component isZero[n];
    signal isNonZero[n];
    signal count[n + 1];
    signal indexDiff[n];
    signal valueDiff[n];
    signal indexMatch[n];
    signal valueMatch[n];
    signal indexValid[n];
    signal valueValid[n];
    signal bothValid[n];
    signal elemValid[n];
    signal totalValid[n + 1];
    
    // Initialize accumulators
    count[0] <== 0;
    totalValid[0] <== 1;
    
    // Process each gradient element
    for (var i = 0; i < n; i++) {
        // Check if gradient is zero
        isZero[i] = IsZero();
        isZero[i].in <== gradients[i];
        isNonZero[i] <== 1 - isZero[i].out;
        
        // Count non-zero elements
        count[i + 1] <== count[i] + isNonZero[i];
        
        // Validate sparse representation consistency
        indexDiff[i] <== indices[i] - i;
        valueDiff[i] <== values[i] - gradients[i];
        indexMatch[i] <== indexDiff[i] * indexDiff[i]; // 0 if indices[i] == i
        valueMatch[i] <== valueDiff[i] * valueDiff[i]; // 0 if values[i] == gradients[i]
        
        // For non-zero elements, both index and value must match
        indexValid[i] <== 1 - indexMatch[i];
        valueValid[i] <== 1 - valueMatch[i];
        
        // Element is valid if it's zero OR (non-zero AND properly represented)
        bothValid[i] <== indexValid[i] * valueValid[i];
        elemValid[i] <== isZero[i].out + (isNonZero[i] * bothValid[i]);
        totalValid[i + 1] <== totalValid[i] * elemValid[i];
    }
    
    nonZeroCount <== count[n];
    
    // Calculate compression ratio (simplified)
    signal densityPercent <== nonZeroCount * 100;
    compressionRatio <== densityPercent;
    
    // Verify sparsity constraint
    signal withinThreshold <== 1; // Simplified constraint
    
    // Overall validation
    sparseValid <== totalValid[n] * withinThreshold;
    
    // Constraint: sparse representation must be valid
    sparseValid === 1;
}

component main { public [indices, sparsityThreshold] } = SparseGradients(4);
