public class Adam {
    private double[][] mWf, mWi, mWC, mWo, mbf, mbi, mbC, mbo;
    private double[][] vWf, vWi, vWC, vWo, vbf, vbi, vbC, vbo;

    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private final double learningRate = 0.001;
    private int t = 0;

    public Adam(int hidden, int input) {
        this.mWf = NumJava.zeros(hidden, hidden + input);
        this.mWi = NumJava.zeros(hidden, hidden + input);
        this.mWC = NumJava.zeros(hidden, hidden + input);
        this.mWo = NumJava.zeros(hidden, hidden + input);

        this.vWf = NumJava.zeros(hidden, hidden + input);
        this.vWi = NumJava.zeros(hidden, hidden + input);
        this.vWC = NumJava.zeros(hidden, hidden + input);
        this.vWo = NumJava.zeros(hidden, hidden + input);

        this.mbf = NumJava.zeros(hidden, 1);
        this.mbi = NumJava.zeros(hidden, 1);
        this.mbC = NumJava.zeros(hidden, 1);
        this.mbo = NumJava.zeros(hidden, 1);

        this.vbf = NumJava.zeros(hidden, 1);
        this.vbi = NumJava.zeros(hidden, 1);
        this.vbC = NumJava.zeros(hidden, 1);
        this.vbo = NumJava.zeros(hidden, 1);
    }

    public void update(double[][] W, double[][] dW, double[][] m, double[][] v) {
        t += 1;

        m = NumJava.add(NumJava.times(beta1, m), NumJava.times(1-beta1, dW));
        v = NumJava.add(NumJava.times(beta2, v), NumJava.times(1-beta2, NumJava.times(dW, dW)));

        double[][] mHat = NumJava.div(m, 1 - Math.pow(beta1, t));
        double[][] vHat = NumJava.div(v, 1 - Math.pow(beta2, t));

        W = NumJava.subtract(W, NumJava.times(learningRate, NumJava.div(NumJava.add(NumJava.sqrt(vHat), epsilon), mHat)));
    }

    public void apply(LSTM lstm, double[][] dWf, double[][] dWi, double[][] dWC, double[][] dWo,
                                 double[][] dbf, double[][] dbi, double[][] dbC, double[][] dbo) {
        update(lstm.Wf, dWf, mWf, vWf);
        update(lstm.Wi, dWi, mWi, vWi);
        update(lstm.WC, dWC, mWC, vWC);
        update(lstm.Wo, dWo, mWo, vWo);

        update(lstm.bf, dbf, mbf, vbf);
        update(lstm.bi, dbi, mbi, vbi);
        update(lstm.bC, dbC, mbC, vbC);
        update(lstm.bo, dbo, mbo, vbo);
    }
}