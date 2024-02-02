/*
** EPITECH PROJECT, 2019
** my_pool
** File description:
** function of pool
*/

#include <unistd.h>

void my_putchar(char c)
{
    write (1, &c, 1);
}

int my_putstr(char const *str)
{
    for (; *str != '\0' && str != 0; ++str)
        write(1, str, 1);
    return (0);
}

void print(long nb)
{
    if (nb >= 10)
        print(nb / 10);
    if (nb > 0)
        my_putchar((nb % 10) + 48);
}

int my_putnbr(long nb)
{
    if (nb == 0) {
        write(1, "0", 1);
        return (0);
    }
    if (nb < 0) {
        write(1, "-", 1);
        nb = -nb;
    }
    print(nb);
    return (0);
}

int my_strlen(char const *str)
{
    int i = 0;

    for (i = 0; str[i] != '\0'; i++);
    return (i);
}