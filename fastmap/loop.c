#include <stdio.h>
#define N 4

void print(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void foreach_permutation(void fn(int*, int), int arr[], int n, int k) {
    if (k == 1) {
        fn(arr, n);
    } else {
        foreach_permutation(fn, arr, n, k - 1);

        for (int i = 0; i < k - 1; i++) {
            if (k % 2 == 0) {
                int el = arr[i];
                arr[i] = arr[k - 1];
                arr[k - 1] = el;
            } else {
                int el = arr[0];
                arr[0] = arr[k - 1];
                arr[k - 1] = el;
            }

            foreach_permutation(fn, arr, n, k - 1);
        }
    }
}

int main() {
    int arr[N] = {0};
    for (int i = 0; i < N; i++) arr[i] = i;
    foreach_permutation(print, arr, N, N);
    return 0;
}