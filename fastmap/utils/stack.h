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
    StackNode *head;
} Stack;

Stack *
stack_alloc ()
{
    Stack *stack = malloc (sizeof (Stack));
    stack->size = 0;
    stack->head = NULL;
    return stack;
}

void
push (Stack *stack, void *item)
{
    StackNode *node = malloc (sizeof (StackNode));
    node->item = item;
    node->next = stack->head;
    stack->head = node;
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

    void *item = stack->head->item;
    StackNode *tmp = stack->head;
    stack->head = stack->head->next;
    stack->size--;
    free (tmp);
    return item;
}
