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

int iaupdate(char **map, int line, int nbr, int z)
{
    int right = (z * 2) - 1;
    int temp = nbr;

    my_putstr("IA's turn...\nIA removed ");
    my_put_nbr(nbr);
    my_putstr(" match(es) from line ");
    my_put_nbr(line);
    my_putchar('\n');
    for (int i = right; temp != 0; --i)
        if (map[line][i] == '|') {
            map[line][i] = ' ';
            temp--;
        }
    for (int i = 0; i <= z + 1; ++i) {
        my_putstr(map[i]);
        my_putchar('\n');
    }
}

int update(char **map, int line, int nbr, int z)
{
    int right = (z * 2) - 1;
    int temp = nbr;

    my_putstr("Player removed ");
    my_put_nbr(nbr);
    my_putstr(" match(es) from line ");
    my_put_nbr(line);
    my_putchar('\n');
    for (int i = right; temp != 0; --i)
        if (map[line][i] == '|') {
            map[line][i] = ' ';
            temp--;
        }
    for (int i = 0; i <= z + 1; ++i) {
        my_putstr(map[i]);
        my_putchar('\n');
    }
}

int checkupdate(char **map, int line, int nbr, int z)
{
    int count = 0;
    for (int i = 0; map[line][i]; ++i)
        if (map[line][i] == '|')
            ++count;
    if (count < nbr) {
        my_putstr("Error: not enought matches on this line\n");
        return (101);
    }
    update(map, line, nbr, z);
}