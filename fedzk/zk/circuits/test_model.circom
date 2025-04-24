pragma circom 2.0.0;

template ModelUpdate(n) {
    signal input gradients[n];
    signal output norm;
    var sum = 0;
    for (var i = 0; i < n; i++) {
        sum += gradients[i] * gradients[i];
    }
    norm <== sum;
}

component main { public [gradients] } = ModelUpdate(4);
