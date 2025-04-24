pragma circom 2.0.0;

template MinimalCircuit() {
    signal output out;
    out <== 1;
}

component main = MinimalCircuit();
