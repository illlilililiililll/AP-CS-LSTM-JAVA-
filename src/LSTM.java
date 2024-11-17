public class LSTM {
    protected double[][] Wf, Wi, WC, Wo; // 가중치 행렬 선언
    protected double[][] bf, bi, bC, bo; // 편향 행렬 선언

    private Adam adam; // Adam optimizer 선언

    public LSTM(int input, int hidden) {
        // 가중치 행렬 초기화
        this.Wf = NumJava.randn(hidden, hidden + input);
        this.Wi = NumJava.randn(hidden, hidden + input);
        this.WC = NumJava.randn(hidden, hidden + input);
        this.Wo = NumJava.randn(hidden, hidden + input);
        // 편향 행렬 초기화
        this.bf = NumJava.zeros(hidden, 1);
        this.bi = NumJava.zeros(hidden, 1);
        this.bC = NumJava.zeros(hidden, 1);
        this.bo = NumJava.zeros(hidden, 1);
        // Adam 초기화
        this.adam = new Adam(hidden, input);
    }
    
    // 순방향 전파
    double[][][] forward(double[][] x, double[][] h_prev, double[][] C_prev) {
        double[][] concat = NumJava.vstack(h_prev, x);

        // Forget Gate
        double[][] f_t = NumJava.sigmoid(NumJava.add(NumJava.dot(Wf, concat), bf));

        // Input Gate
        double[][] i_t = NumJava.sigmoid(NumJava.add(NumJava.dot(Wi, concat), bi));

        // Cell State Candidate
        double[][] C_tilda = NumJava.tanh(NumJava.add(NumJava.dot(WC, concat), bC));

        // Cell State
        double[][] C_t = NumJava.add(NumJava.times(f_t, C_prev), NumJava.times(i_t, C_tilda));

        // Output Gate
        double[][] o_t = NumJava.sigmoid(NumJava.add(NumJava.dot(Wo, concat), bo));

        // Hidden State
        double[][] h_t = NumJava.times(o_t, NumJava.tanh(C_t));

        return new double[][][] { f_t, i_t, C_tilda, o_t, C_t, h_t };
    }

    // 역방향 전파
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
        // 오차역전파를 통한 편미분 계산
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

        // 가중치 미분 정의
        double[][] T = NumJava.transpose(concat);
        double[][] dWf = NumJava.dot(df_t, T);
        double[][] dWi = NumJava.dot(di_t, T);
        double[][] dWC = NumJava.dot(dC_tilda, T);
        double[][] dWo = NumJava.dot(do_t, T);

        double[][] dbf = NumJava.sum(df_t, 1);
        double[][] dbi = NumJava.sum(di_t, 1);
        double[][] dbC = NumJava.sum(dC_tilda, 1);
        double[][] dbo = NumJava.sum(do_t, 1);

        double[][] dh_prev = NumJava.dot(NumJava.transpose(Wf), df_t);
        dh_prev = NumJava.add(dh_prev, NumJava.dot(NumJava.transpose(Wi), di_t));
        dh_prev = NumJava.add(dh_prev, NumJava.dot(NumJava.transpose(WC), dC_tilda));
        dh_prev = NumJava.add(dh_prev, NumJava.dot(NumJava.transpose(Wo), do_t));

        double[][] dC_prev = NumJava.times(dC_t, f_t);


        return new double[][][] {dh_prev, dC_prev, dWf, dWi, dWC, dWo, dbf, dbi, dbC, dbo};
    }
    
    // 모델 학습
    public void fit(double[][][] X, double[][][] Y, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) { // epoch (반복횟수) 만큼 반복
            double loss = 0.0; // 오차

            // 이전 Hidden State, Cell State 초기화
            double[][] h_prev = NumJava.zeros(Wf.length, 1);
            double[][] C_prev = NumJava.zeros(WC.length, 1);

            for (int t = 0; t < X.length; t++) { // Time Step (X 행렬 행) 만큼 반복
                double[][] x = X[t];
                double[][] y_real = Y[t];

                // 순전파
                double[][][] fwd = forward(x, h_prev, C_prev);

                double[][] f_t = fwd[0];
                double[][] i_t = fwd[1];
                double[][] C_tilda = fwd[2];
                double[][] o_t = fwd[3];
                double[][] C_t = fwd[4];
                double[][] h_t = fwd[5];

                double[][] pred = NumJava.softmax(h_t); // SoftMax 적용

                loss += NumJava.crossEntropyLoss(pred, y_real); // CrossEntropyLoss 오차

                // 다음 Hidden State, Cell State 정의
                double[][] dh_next = NumJava.subtract(pred, y_real);
                double[][] dC_next = NumJava.zeros(C_t.length, C_t[0].length);

                // 역전파
                double[][][] backProp = backward(dh_next, dC_next, C_prev, f_t, i_t, C_tilda, o_t, C_t, h_prev, x);

                double[][] dh_prev = backProp[0];
                double[][] dC_prev = backProp[1];
                double[][] dWf = backProp[2];
                double[][] dWi = backProp[3];
                double[][] dWC = backProp[4];
                double[][] dWo = backProp[5];
                double[][] dbf = backProp[6];
                double[][] dbi = backProp[7];
                double[][] dbC = backProp[8];
                double[][] dbo = backProp[9];

                // 가중치 업데이트
                adam.apply(this, dWf, dWi, dWC, dWo, dbf, dbi, dbC, dbo);

                h_prev = h_t;
                C_prev = C_t;
            }

            System.out.println("# Epoch " + (epoch+1) + "/" + epochs + "\t(Loss: " + loss/X.length + ")");
        }
    }
}