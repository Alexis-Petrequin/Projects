/*
** EPITECH PROJECT, 2019
** my_putstr
** File description:
** my_putstr.c
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "include.h"

char **lines(int z, char **map, int nb, int sticks)
{
    static int nbr = 0;
    int i = 0;
    map[nb][i] = '*';
    ++i;
    for (int temp = 0; temp < nbr; ++temp, ++i)
        map[nb][i] = ' ';
    for (int temp = 0; temp < sticks; ++temp, ++i)
        map[nb][i] = '|';
    for (int temp = 0; temp < nbr; ++temp, ++i)
        map[nb][i] = ' ';
    map[nb][i] = '*';
    ++nbr;
    return (map);
}

char **lineone(int z, char **map, int nb)
{
    for (int i = 0; i <= (z * 2); ++i)
        map[nb][i] = '*';
    return (map);
}

char **drawmap(int z, char **map)
{
    int i = (z + 1);
    int sticks = (z * 2) + 1;

    for (int count = 0; i >= 0; --i, ++count) {
        if (i == (z + 1))
            map =lineone(z, map, i);
        else if (i == 0)
            map = lineone(z, map, i);
        else
            map = lines(z, map, i, sticks);
        sticks -= 2;
    }
    for (int i = 0; i <= z + 1; ++i){
        my_putstr(map[i]);
        my_putchar('\n');
    }
    my_putchar('\n');
    return (map);
}

char **create_map(char **av)
{
    int z = number(av[1]);
    char **map = malloc(sizeof(char*) * (z + 2));

    for (int i = 0; i <= (z + 2); ++i)
        map[i] = malloc(sizeof(char) * ((z * 2) + 2));
    map = drawmap(z, map);
    return (map);
}
