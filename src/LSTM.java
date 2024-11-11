public class LSTM {
    private double[][] Wf;
    private double[][] Wi;
    private double[][] WC;
    private double[][] Wo;

    private double[][] bf;
    private double[][] bi;
    private double[][] bC;
    private double[][] bo;

    public LSTM(int input, int hidden) {
        this.Wf = NumJava.randn(hidden, hidden + input);
        this.Wi = NumJava.randn(hidden, hidden + input);
        this.WC = NumJava.randn(hidden, hidden + input);
        this.Wo = NumJava.randn(hidden, hidden + input);

        this.bf = NumJava.zeros(hidden, 1);
        this.bi = NumJava.zeros(hidden, 1);
        this.bC = NumJava.zeros(hidden, 1);
        this.bo = NumJava.zeros(hidden, 1);
    }

    double[][][] forward(double[][] x, double[][] h_prev, double[][] C_prev) {
        double[][] concat = NumJava.vstack(h_prev, x);

        double[][] f_t = NumJava.sigmoid(NumJava.add(NumJava.dot(Wf, concat), bf));

        double[][] i_t = NumJava.sigmoid(NumJava.add(NumJava.dot(Wi, concat), bi));
        double[][] g_t = NumJava.tanh(NumJava.add(NumJava.dot(WC, concat), bC));

        double[][] C_t = NumJava.add(NumJava.times(f_t, C_prev), NumJava.times(i_t, g_t));

        double[][] o_t = NumJava.sigmoid(NumJava.add(NumJava.dot(Wo, concat), bo));

        double[][] h_t = NumJava.times(o_t, NumJava.tanh(C_t));

        return new double[][][] {h_t, C_t};
    }
}