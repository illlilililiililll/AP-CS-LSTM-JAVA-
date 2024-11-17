public class Adam {
    private double[][] mWf, mWi, mWC, mWo, mbf, mbi, mbC, mbo;
    private double[][] vWf, vWi, vWC, vWo, vbf, vbi, vbC, vbo;

    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private final double learningRate = 0.001;
    private int t = 0;

    public Adam(int hidden, int input) {
        // Initialize moments for weights and biases
        mWf = NumJava.zeros(hidden, hidden + input);
        mWi = NumJava.zeros(hidden, hidden + input);
        mWC = NumJava.zeros(hidden, hidden + input);
        mWo = NumJava.zeros(hidden, hidden + input);

        mbf = NumJava.zeros(hidden, 1);
        mbi = NumJava.zeros(hidden, 1);
        mbC = NumJava.zeros(hidden, 1);
        mbo = NumJava.zeros(hidden, 1);

        vWf = NumJava.zeros(hidden, hidden + input);
        vWi = NumJava.zeros(hidden, hidden + input);
        vWC = NumJava.zeros(hidden, hidden + input);
        vWo = NumJava.zeros(hidden, hidden + input);

        vbf = NumJava.zeros(hidden, 1);
        vbi = NumJava.zeros(hidden, 1);
        vbC = NumJava.zeros(hidden, 1);
        vbo = NumJava.zeros(hidden, 1);
    }

    private double[][] update(double[][] param, double[][] grad, double[][] m, double[][] v) {
        t += 1;

        // Update biased moment estimates
        double[][] newM = NumJava.add(NumJava.times(beta1, m), NumJava.times(1 - beta1, grad));
        double[][] newV = NumJava.add(NumJava.times(beta2, v), NumJava.times(1 - beta2, NumJava.times(grad, grad)));

        // Bias correction
        double[][] mHat = NumJava.div(newM, 1 - Math.pow(beta1, t));
        double[][] vHat = NumJava.div(newV, 1 - Math.pow(beta2, t));

        return NumJava.subtract(param, NumJava.times(learningRate, NumJava.div(mHat, NumJava.add(NumJava.sqrt(vHat), epsilon)))); // 새로운 값을 반환
    }


    public void apply(LSTM lstm, double[][] dWf, double[][] dWi, double[][] dWC, double[][] dWo,
                      double[][] dbf, double[][] dbi, double[][] dbC, double[][] dbo) {
        // Update weights
        lstm.Wf = update(lstm.Wf, dWf, mWf, vWf);
        lstm.Wi = update(lstm.Wi, dWi, mWi, vWi);
        lstm.WC = update(lstm.WC, dWC, mWC, vWC);
        lstm.Wo = update(lstm.Wo, dWo, mWo, vWo);

        // Update biases
        lstm.bf = update(lstm.bf, dbf, mbf, vbf);
        lstm.bi = update(lstm.bi, dbi, mbi, vbi);
        lstm.bC = update(lstm.bC, dbC, mbC, vbC);
        lstm.bo = update(lstm.bo, dbo, mbo, vbo);
    }
}
