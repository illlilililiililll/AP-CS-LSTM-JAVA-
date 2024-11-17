public class Main {
    public static void main(String[] args) {
        int input = 3;
        int hidden = 5;
        LSTM lstm = new LSTM(input, hidden);

        double[][][] X = new double[3][input][1];
        for (int i = 0; i < 3; i++)
            X[i] = NumJava.randn(input, 1);

        double[][][] Y = {{{1}, {0}, {0}, {0}, {0}}, {{0}, {0}, {1}, {0}, {0}}, {{0}, {0}, {0}, {0}, {1}}};

        lstm.fit(X, Y, 100);
    }
}