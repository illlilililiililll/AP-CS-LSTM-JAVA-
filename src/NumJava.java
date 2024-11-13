import java.util.Arrays;
import java.util.Random;

public class NumJava {
    public static int depth(Object arr) {
        int depth = 0;
        Class<?> type = arr.getClass();
        while (type.isArray()) {
            depth++;
            type = type.getComponentType();
        }
        return depth;
    }

    public static int[] shape(Object arr) {
        int d = depth(arr);
        int[] shape = new int[d];
        Object current = arr;

        for (int i = 0; i < d; i++) {
            shape[i] = java.lang.reflect.Array.getLength(current);
            if (shape[i] > 0)
                current = java.lang.reflect.Array.get(current, 0);
        }
        return shape;
    }

    public static double[][] zeros(int m, int n) {
        double[][] arr = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                arr[i][j] = 0;

        return arr;
    }

    public static double[][] zeros(int m) {
        double[][] arr = new double[m][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                arr[i][j] = 0;

        return arr;
    }

    public static double[][] ones(int m, int n) {
        double[][] arr = new double[m][n];

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                arr[i][j] = 1;

        return arr;
    }

    public static double[][] ones(int m) {
        double[][] arr = new double[m][m];

        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                arr[i][j] = 1;

        return arr;
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double[][] sigmoid(double[][] x) {
        int row = x.length;
        int col = x[0].length;

        double[][] array = new double[row][col];
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                array[i][j] = sigmoid(x[i][j]);

        return array;
    }

    public static double dsigmoid(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static double[][] dsigmoid(double[][] x) {
        int row = x.length;
        int col = x[0].length;

        double[][] array = new double[row][col];
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                array[i][j] = dsigmoid(x[i][j]);

        return array;
    }

    public static double tanh(double x) {
        return (Math.exp(x) - Math.exp(-x))/(Math.exp(x) + Math.exp(-x));
    }

    public static double[][] tanh(double[][] x) {
        int row = x.length;
        int col = x[0].length;

        double[][] array = new double[row][col];
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                array[i][j] = tanh(x[i][j]);

        return array;
    }

    public static double dtanh(double x) {
        return 1 - tanh(x) * tanh(x);
    }

    public static double[][] dtanh(double[][] x) {
        int row = x.length;
        int col = x[0].length;

        double[][] array = new double[row][col];
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                array[i][j] = dtanh(x[i][j]);

        return array;
    }

    public static long power(long x, long n) {
        long result = 1;
        while (n != 0) {
            if (n % 2 == 1)
                result *= x;
            x *= x;
            n /= 2;
        }

        return result;
    }

    public static double[] softmax(double[] x) {
        double[] arr = new double[x.length];
        double temp = 0;
        double sum = 0.0;

        for (int i = 0; i < x.length; i++) {
            temp = Math.exp(x[i] + 1e-12);
            arr[i] = temp;
            sum += temp;
        }

        for (int i = 0; i < x.length; i++)
            arr[i] /= sum;

        return arr;
    }

    public static double[][] softmax(double[][] x) {
        double[][] arr = new double[x.length][x[0].length];
        double temp = 0;
        double sum = 0.0;

        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x[0].length; j++) {
                temp = Math.exp(x[i][j] + 1e-12);
                arr[i][j] = temp;
                sum += temp;
            }

        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x[0].length; j++)
                arr[i][j] /= sum;

        return arr;
    }

    public static double[][] randn(int m, int n) {
        double[][] arr = new double[m][n];

        Random rand = new Random();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                arr[i][j] = rand.nextGaussian();

        return arr;
    }

    public static double[][] randn(int n) {
        double[][] arr = new double[1][n];

        Random rand = new Random();
        for (int i = 0; i < n; i++)
            arr[0][i] = rand.nextGaussian();

        return arr;
    }

    public static int[][] vstack(int[][] A, int[][] B) {
        int Arow = A.length;
        int Acol = A[0].length;
        int Brow = B.length;
        int Bcol = B[0].length;
        int result = Arow + Brow;

        if (Acol != Bcol) throw new IllegalArgumentException("Two matrix must have same columns\nA (" + Arow + ", " + Acol + ")\nB (" + Brow + ", " + Bcol + ")");
        int[][] arr = new int[result][Acol];

        int i;
        for (i = 0; i < Arow; i++)
            System.arraycopy(A[i], 0, arr[i], 0, Acol);

        for (; i < result; i++)
            System.arraycopy(B[i - Arow], 0, arr[i], 0, Acol);

        return arr;
    }

    public static double[][] vstack(double[][] A, double[][] B) {
        int Arow = A.length;
        int Acol = A[0].length;
        int Brow = B.length;
        int Bcol = B[0].length;
        int result = Arow + Brow;

        if (Acol != Bcol) throw new IllegalArgumentException("Two matrix must have same columns\nA (" + Arow + ", " + Acol + ")\nB (" + Brow + ", " + Bcol + ")");
        double[][] arr = new double[result][Acol];

        int i;
        for (i = 0; i < Arow; i++)
            System.arraycopy(A[i], 0, arr[i], 0, Acol);

        for (; i < result; i++)
            System.arraycopy(B[i - Arow], 0, arr[i], 0, Acol);

        return arr;
    }

    public static double[][] add(double[][] A, double[][] B) {
        int[] A_shape = shape(A);
        int[] B_shape = shape(B);

        if (!Arrays.equals(A_shape, B_shape)) {
//            print(A_shape);
//            print(B_shape);
            throw new IllegalArgumentException("Two matrix must have same shape\nA (" + A_shape[0] + ", " + A_shape[1] + ")\nB (" + B_shape[0] + ", " + B_shape[1] + ")");
        }
        double[][] result = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                result[i][j] = A[i][j] + B[i][j];

        return result;
    }

    public static int[][] add(int[][] A, int[][] B) {
        int[] A_shape = shape(A);
        int[] B_shape = shape(B);

        if (!Arrays.equals(A_shape, B_shape)) throw new IllegalArgumentException("Two matrix must have same shape\nA (" + A_shape[0] + ", " + A_shape[1] + ")\nB (" + B_shape[0] + ", " + B_shape[1] + ")");
        int[][] result = new int[A.length][A[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                result[i][j] = A[i][j] + B[i][j];

        return result;
    }

    public static double[][] add(double[][] A, double k) {
        double[][] arr = new double[A.length][A[0].length];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                arr[i][j] = A[i][j] + k;

        return arr;
    }

    public static double[][] subtract(double[][] A, double[][] B) {
        if (!Arrays.equals(shape(A), shape(B))) throw new IllegalArgumentException("Two matrix must have same shape");
        double[][] result = new double[A.length][A[0].length];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                result[i][j] = A[i][j] - B[i][j];

        return result;
    }

    public static double[][] dot(double[][] A, double[][] B) {
        int[] A_shape = shape(A);
        int[] B_shape = shape(B);

        if (A_shape[1] != B_shape[0]) throw new IllegalArgumentException("A column did not match B row");
        double[][] result = new double[A_shape[0]][B_shape[1]];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < B[0].length; j++)
                for (int k = 0; k < A[0].length; k++)
                    result[i][j] += A[i][k] * B[k][j];

        return result;
    }

    public static double[][] times(double[][] A, double[][] B) {
        int[] A_shape = shape(A);
        int[] B_shape = shape(B);

        if (!Arrays.equals(A_shape, B_shape)) throw new IllegalArgumentException("Two matrix must have same size\nA:" + A.length + ", " + A[0].length + "\nB:" + B.length + ", " + B[0].length);
        double[][] result = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                result[i][j] = A[i][j] * B[i][j];

        return result;
    }

    public static double[][] times(double k, double[][] A) {
        double[][] arr = new double[A.length][A[0].length];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                arr[i][j] = A[i][j] * k;

        return arr;
    }

    public static double[][] div(double[][] A, double k) {
        double[][] arr = new double[A.length][A[0].length];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                arr[i][j] = A[i][j] / k;

        return arr;
    }

    public static double[][] div(double[][] A, double[][] B) {
        double[][] arr = new double[A.length][A[0].length];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                arr[i][j] = A[i][j] / B[i][j];

        return arr;
    }

    public static double[][] sqrt(double[][] A) {
        double[][] arr = new double[A.length][A[0].length];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                arr[i][j] = Math.sqrt(A[i][j]);

        return arr;
    }

    public static double[][] sum(double[][] A, int axis) {
        double[][] arr = new double[A.length][A[0].length];
        if (axis == 0) {
            arr = new double[1][A[0].length];

            for (double[] doubles : A)
                for (int j = 0; j < A[0].length; j++)
                    arr[0][j] += doubles[j];
        }
        else if (axis == 1) {
            arr = new double[A.length][1];
            for (int i = 0; i < A.length; i++)
                for (int j = 0; j < A[0].length; j++)
                    arr[i][0] += A[i][j];
        }

        return arr;
    }

    public static double[][] transpose(double[][] A) {
        double[][] arr = new double[A[0].length][A.length];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                arr[j][i] = A[i][j];

        return arr;
    }

    public static void print(double[][] A) {
        System.out.println("{ ");
        for (double[] arr : A) {
            for (int j = 0; j < A[0].length; j++)
                System.out.print(arr[j] + " ");
            System.out.println();
        }
        System.out.println("}");
    }

    public static void print(double[] A) {
        System.out.print("{ ");
        for (double i : A) {
            System.out.print(i + " ");
        }
        System.out.println("}");
    }

    public static void print(int[] A) {
        for (int i : A) {
            System.out.print(i + " ");
        }
        System.out.println();
    }

    public static double crossEntropyLoss(double[][] pred, double[][] real) {
        print(shape(pred));
        print(shape(real));
        double loss = 0.0;
        for (int i = 0; i < real.length; i++)
            for (int j = 0; j < real[0].length; j++)
                loss -= real[i][j] * Math.log(pred[i][j] + 1e-8);

        return loss;
    }
}
