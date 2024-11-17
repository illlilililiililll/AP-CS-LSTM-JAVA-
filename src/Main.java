public class Main {
    public static void main(String[] args) {
        int input = 3;
        int hidden = 5;
        LSTM lstm = new LSTM(input, hidden);

        double[][] x_t = NumJava.randn(input, 1);
        double[][] y_t = {{1}, {0}, {0}};
        NumJava.print(NumJava.shape(x_t));
        NumJava.print(NumJava.shape(y_t));

        double[][] h_prev = NumJava.zeros(hidden, 1);
        double[][] C_prev = NumJava.zeros(hidden, 1);

        double[][][] result = lstm.forward(x_t, h_prev, C_prev);
        double[][] f_t, i_t, C_tilda, o_t, C_t, h_t;
        NumJava.print(result);

        double loss = 0.0;
        f_t = result[0];
        i_t = result[1];
        C_tilda = result[2];
        o_t = result[3];
        C_t = result[4];
        h_t = result[5];

        double[][] pred = NumJava.softmax(h_t);
        NumJava.print(NumJava.shape(pred));
        double loss_cur = NumJava.crossEntropyLoss(pred, y_t);
//        System.out.println(loss_cur);
        loss += loss_cur;

        NumJava.print(NumJava.shape(y_t));
        double[][] dh_next = NumJava.subtract(pred, y_t); // 문제의 원흉

        double[][] dC_next = NumJava.zeros(C_t.length, C_t[0].length);

        double[][][] backProp = lstm.backward(dh_next, dC_next, C_prev, f_t, i_t, C_tilda, o_t, C_t, h_prev, x_t);
    }
}