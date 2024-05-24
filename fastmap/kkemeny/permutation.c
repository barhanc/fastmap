#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void generate_next_permutation(int *arr, int n, int pos) {
    int i;
    for (i = n - 1; i > pos; i--) {
        if (arr[i] > arr[pos]) {
            break;
        }
    }
    swap(&arr[pos], &arr[i]);
}

void generate_all_permutations(int *arr, int n) {
    printf("Initial permutation: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    while (1) {
        int pos = -1;
        for (int i = n - 2; i >= 0; i--) {
            if (arr[i] < arr[i + 1]) {
                pos = i;
                break;
            }
        }

        if (pos == -1) {
            break;
        }

        generate_next_permutation(arr, n, pos);

        int left = pos + 1, right = n - 1;
        while (left < right) {
            swap(&arr[left], &arr[right]);
            left++;
            right--;
        }

        printf("Next permutation: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
}

int main() {
    int n;
    printf("Enter the size of the permutation: ");
    scanf("%d", &n);

    int *arr = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        arr[i] = i + 1;
    }

    generate_all_permutations(arr, n);

    free(arr);

    return 0;
}
