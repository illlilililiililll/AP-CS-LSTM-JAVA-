public class Main {
    public static void main(String[] args) {
        int inputSize = 3;   // 입력 크기
        int hiddenSize = 5;  // 은닉 상태 크기
        LSTM lstm = new LSTM(inputSize, hiddenSize);

        // 예제 입력 데이터 (초기 입력, 은닉 상태 및 셀 상태)
        double[][] x = NumJava.randn(inputSize, 1);  // 입력 데이터 (inputSize x 1)
        double[][] h_prev = NumJava.zeros(hiddenSize, 1); // 초기 은닉 상태
        double[][] C_prev = NumJava.zeros(hiddenSize, 1); // 초기 셀 상태

        // forwardProp 호출
        double[][][] result = lstm.forward(x, h_prev, C_prev);

        NumJava.print(x);
//        NumJava.print(NumJava.shape(x));

        // 결과 출력
        System.out.println("Hidden state (h_t):");
        NumJava.print(result[0]);  // h_t 출력

        System.out.println("Cell state (C_t):");
        NumJava.print(result[1]);  // C_t 출력
    }
}