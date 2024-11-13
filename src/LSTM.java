public class LSTM {
    protected double[][] Wf, Wi, WC, Wo;
    protected double[][] bf, bi, bC, bo;

    protected double[][] dWf, dWi, dWC, dWo;
    protected double[][] dbf, dbi, dbC, dbo;

    private Adam adam;

    public LSTM(int input, int hidden) {
        this.Wf = NumJava.randn(hidden, hidden + input);
        this.Wi = NumJava.randn(hidden, hidden + input);
        this.WC = NumJava.randn(hidden, hidden + input);
        this.Wo = NumJava.randn(hidden, hidden + input);

        this.bf = NumJava.zeros(hidden, 1);
        this.bi = NumJava.zeros(hidden, 1);
        this.bC = NumJava.zeros(hidden, 1);
        this.bo = NumJava.zeros(hidden, 1);

        this.adam = new Adam(hidden, input);
    }

    double[][][] forward(double[][] x, double[][] h_prev, double[][] C_prev) {
        double[][] concat = NumJava.vstack(h_prev, x);

        double[][] f_t = NumJava.sigmoid(NumJava.add(NumJava.dot(Wf, concat), bf));

        double[][] i_t = NumJava.sigmoid(NumJava.add(NumJava.dot(Wi, concat), bi));
        double[][] C_tilda = NumJava.tanh(NumJava.add(NumJava.dot(WC, concat), bC));

        double[][] C_t = NumJava.add(NumJava.times(f_t, C_prev), NumJava.times(i_t, C_tilda));

        double[][] o_t = NumJava.sigmoid(NumJava.add(NumJava.dot(Wo, concat), bo));

        double[][] h_t = NumJava.times(o_t, NumJava.tanh(C_t));

        return new double[][][] { h_t, C_t };
    }

    double[][][] backward(
            double[][] dh_next, // ∂L/∂h_t
            double[][] dC_next,
            double[][] C_prev,
            double[][] f_t,
            double[][] i_t,
            double[][] C_tilda,
            double[][] o_t,
            double[][] C_t,
            double[][] h_prev,
            double[][] x
    ) {
        // Output Gate
        double[][] do_t = NumJava.times(NumJava.times(dh_next, NumJava.tanh(C_t)), NumJava.dsigmoid(o_t));

        // Cell State
        double[][] dC_t = NumJava.add(NumJava.times(dh_next, NumJava.times(o_t, NumJava.dtanh(C_t))), NumJava.times(dC_next, f_t));

        // Input Gate
        double[][] di_t = NumJava.times(NumJava.times(dh_next, C_tilda), NumJava.dsigmoid(i_t));

        // Cell State Candidate
        double[][] dC_tilda = NumJava.times(NumJava.times(dC_t, i_t), NumJava.dtanh(C_tilda));

        // Forget Gate
        double[][] df_t = NumJava.times(NumJava.times(dC_t, C_prev), NumJava.dsigmoid(f_t));

        double[][] concat = NumJava.vstack(h_prev, x);

        double[][] T = NumJava.transpose(concat);
        this.dWf = NumJava.dot(df_t, T);
        this.dWi = NumJava.dot(di_t, T);
        this.dWC = NumJava.dot(dC_tilda, T);
        this.dWo = NumJava.dot(do_t, T);

        this.dbf = df_t;
        this.dbi = di_t;
        this.dbC = dC_tilda;
        this.dbo = do_t;

        double[][] dh_prev = NumJava.dot(NumJava.transpose(Wf), df_t);
        dh_prev = NumJava.add(dh_prev, NumJava.dot(NumJava.transpose(Wi), di_t));
        dh_prev = NumJava.add(dh_prev, NumJava.dot(NumJava.transpose(WC), dC_tilda));
        dh_prev = NumJava.add(dh_prev, NumJava.dot(NumJava.transpose(Wo), do_t));

        double[][] dC_prev = NumJava.times(dC_t, f_t);


        return new double[][][] { dh_prev, dC_prev };
    }

    void update() {

    }
}