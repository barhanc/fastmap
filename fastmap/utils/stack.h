/**
 * @file queue.h
 * @brief Simple implementation of a stack or Last-In-First-Out (LIFO) queue.
 *
 * This file contains a basic implementation of a generic stack using a linked list. The stack
 * allows for elements to be added to and removed from the top.
 *
 * NOTE: This implementation is not thread-safe.
 *
 * Public fields:
 * - `size`: number of elements on the stack
 *
 * Public functions:
 * - `Queue *stack_alloc()`: allocate memory for the stack and return pointer to it
 * - `void push(Stack *s, void *item)`: push a generic item on the top of the stack
 * - `void *pop(Stack *s)`: remove and return the generic item from the top of the stack
 */

#include <stdio.h>
#include <stdlib.h>

typedef struct StackNode
{
    void *item;
    struct StackNode *next;
} StackNode;

typedef struct Stack
{
    size_t size;
    StackNode *top;
} Stack;

Stack *
stack_alloc ()
{
    Stack *stack = malloc (sizeof (Stack));
    stack->size = 0;
    stack->top = NULL;
    return stack;
}

void
push (Stack *stack, void *item)
{
    StackNode *node = malloc (sizeof (StackNode));
    node->item = item;
    node->next = stack->top;
    stack->top = node;
    stack->size++;
    return;
}

void *
pop (Stack *stack)
{
    if (stack->size == 0)
    {
        printf ("ERROR: Trying to pop an empty stack!\n");
        return NULL;
    }

    void *item = stack->top->item;
    StackNode *tmp = stack->top;
    stack->top = stack->top->next;
    stack->size--;
    free (tmp);

    return item;
}
