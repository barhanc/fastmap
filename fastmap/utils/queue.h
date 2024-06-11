/**
 * @file queue.h
 * @brief Simple implementation of a First-In-First-Out (FIFO) queue.
 *
 * This file contains a basic implementation of a generic FIFO queue using a linked list. The FIFO
 * queue allows for elements to be added to the end and removed from the front, following the
 * first-in-first-out principle.
 *
 * NOTE: This implementation is not thread-safe.
 *
 * Public fields:
 * - `size`: number of elements in the queue
 *
 * Public functions:
 * - `Queue *queue_alloc()`: allocate memory for the queue and return pointer to it
 * - `void enqueue(Queue *q, void *item)`: add a generic item to the end of the queue
 * - `void *dequeue(Queue *q)`: remove and return the generic item from the front of the queue
 *
 * Example:
 * ```
 * #include "queue.h"
 *
 * typedef struct Data{ int val; } Data;
 *
 * int main()
 * {
 *     Queue *q = queue_alloc ();
 *
 *     Data *data1 = malloc(sizeof(Data)), *data2 = malloc(sizeof(Data));
 *     data1->val = 42, data2->val = 404;
 *
 *     enqueue(q, (void*)data1);
 *     enqueue(q, (void*)data2);
 *
 *     Data *data;
 *
 *     data = (Data*)dequeue(q);
 *     printf("%d\n", data->val);
 *     free(data);
 *
 *     data = (Data*)dequeue(q);
 *     printf("%d\n", data->val);
 *     free(data);
 *
 *     free(q);
 *     return 0;
 * }
 * ```
 */

#include <stdlib.h>

typedef struct QueueNode
{
    void *item;
    struct QueueNode *next;
} QueueNode;

typedef struct Queue
{
    size_t size;
    QueueNode *head;
    QueueNode *tail;
} Queue;

Queue *
queue_alloc ()
{
    Queue *q = malloc (sizeof (Queue));
    q->size = 0;
    q->head = NULL;
    q->tail = NULL;
    return q;
}

void
enqueue (Queue *q, void *item)
{
    QueueNode *node = malloc (sizeof (QueueNode));
    node->item = item;
    node->next = NULL;

    if (q->size == 0)
    {
        q->head = node;
        q->tail = node;
        q->size = 1;
        return;
    }

    q->tail->next = node;
    q->tail = node;
    q->size++;
    return;
}

void *
dequeue (Queue *q)
{
    if (q->size == 0)
    {
        printf ("ERROR: Trying to dequeue an empty queue!\n");
        return NULL;
    }

    void *item = q->head->item;

    if (q->size == 1)
    {
        free (q->head);
        q->head = NULL;
        q->tail = NULL;
        q->size = 0;
        return item;
    }

    QueueNode *tmp = q->head;
    q->head = q->head->next;
    q->size--;
    free (tmp);
    return item;
}
