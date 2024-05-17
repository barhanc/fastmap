#include <stdio.h>
#include <stdlib.h>

/**
 * @brief TODO:...
 *
 */
typedef struct QueueNode
{
    void *item;
    struct QueueNode *next;
} QueueNode;

/**
 * @brief TODO:...
 *
 */
typedef struct Queue
{
    size_t size;
    QueueNode *head;
    QueueNode *tail;
} Queue;

/**
 * @brief TODO:...
 *
 * @return Queue*
 */
Queue *
queue_alloc ()
{
    Queue *q = malloc (sizeof (Queue));
    q->size = 0;
    q->head = NULL;
    q->tail = NULL;
    return q;
}

/**
 * @brief TODO:...
 *
 * @param q
 * @param item
 */
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

/**
 * @brief TODO:...
 *
 * @param q
 * @return void*
 */
void *
dequeue (Queue *q)
{
    if (q->size == 0)
    {
        free (q);
        printf ("ERROR: Trying to dequeue an empty queue!\n");
        exit (EXIT_FAILURE);
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

/**
 * Example usage
 * ```
 * #include "queue.h"
 * #include <stdio.h>
 * #include <stdlib.h>
 *
 * typedef struct Data
 * {
 *     int val;
 * } Data;
 *
 * int
 * main ()
 * {
 *     size_t n = 10, m = 6;
 *     Queue *q = queue_alloc ();
 *
 *     // Enqueue `n` elements
 *     for (size_t i = 0; i < n; i++)
 *     {
 *         Data *data = malloc (sizeof (Data));
 *         data->val = i;
 *         enqueue (q, (void *)data);
 *     }
 *
 *     // Dequeue `m` elements and print them
 *     for (size_t i = 0; i < m; i++)
 *     {
 *         Data *data = (Data *)dequeue (q);
 *         printf ("%d\n", data->val);
 *         free (data);
 *     }
 *
 *     // Make sure that queue is free before freeing `q`
 *     if (q->size > 0)
 *     {
 *         printf ("Not all elements have been dequeued. Dequeueing them before freeing `q`...\n");
 *         while (q->size > 0)
 *         {
 *             Data *data = (Data *)dequeue (q);
 *             free (data);
 *         }
 *         printf ("Done! Now you can safely free `q`\n");
 *     }
 *
 *     free (q);
 *     return 0;
 * }
 * ```
 */